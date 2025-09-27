from utils import check_mkdir, pseudo_label_sample, compute_window_energy, average_pooling, map_label_after_pooling, \
    batch_transform_saab_on_demand
import numpy as np
import os
from model import Model
import argparse
import gc
from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', type=str, required=True, choices=['p', 's'])
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--wave_ratio', type=float, default=0.2)
    parser.add_argument('--num_sample', type=lambda s: int(float(s)), default='4e6')

    args = parser.parse_args()
    wave = args.wave
    exp_name = args.exp_name
    wave_ratio = args.wave_ratio
    num_sample = args.num_sample
    assert wave == 'p' or wave == 's'

    ckpt_path = './exp'
    save_path = os.path.join(ckpt_path, exp_name, f'{wave}_wave')
    data_path = './data/'

    check_mkdir(save_path)
    logger.add(os.path.join(save_path, 'logfile.log'), enqueue=True, level="INFO")

    # Load the saved arrays
    if wave == 'p':
        X_full = np.load(os.path.join(data_path, "X_train.npy"), mmap_mode='r')
        y_full = np.load(os.path.join(data_path, "P_train.npy")).astype(np.float32)
    else:
        X_full = np.load(os.path.join(data_path, "X_train.npy"), mmap_mode='r')
        y_full = np.load(os.path.join(data_path, "S_train.npy")).astype(np.float32)

    logger.info(f"Train Model for {wave} wave")
    logger.info(f"X_full shape: {X_full.shape}")
    logger.info(f"y_full shape: {y_full.shape}")

    # Training
    win_len_list = {
        4: 33,  # 375
        3: 65,  # 750
        2: 129,  # 1500
        1: 257,  # 3000
        0: 513,  # 6000
    }
    delta_t_list = {
        4: 3,
        3: 6,
        2: 12,
    }
    RFT_dims = {
        4: 80,
        3: 120,
        2: 200,
    }
    saab_kernel_list = {
        4: 8,
        3: 16,
        2: 16
    }
    saab_stride_list = {
        4: 4,
        3: 8,
        2: 8
    }

    N_waves = X_full.shape[0]
    wave_size = int(N_waves * wave_ratio)
    if wave_ratio != 1:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(N_waves, size=wave_size, replace=False))
    else:
        idx = np.arange(N_waves)

    active_levels = sorted(RFT_dims.keys(), reverse=True)

    for level in active_levels:
        logger.warning(f"========== Level {level} ==========")
        win_len = win_len_list[level]
        delta_t = delta_t_list[level]
        saab_kernel = saab_kernel_list[level]
        saab_stride = saab_stride_list[level]

        X_train = X_full[idx]
        y_train = y_full[idx]
        logger.warning(f"Wave ratio: {wave_ratio} from {N_waves}, wave_size: {wave_size}")

        X_pool = average_pooling(X_train, num_transform=level)
        y_pool = map_label_after_pooling(y_train, num_transform=level)
        logger.warning(f"After average pooling X_pool shape: {X_pool.shape}, y_pool shape: {y_pool.shape}")

        del X_train
        gc.collect()

        # Generate pseudo labels and sampling
        pseudo_labels, subsampled_indices = pseudo_label_sample(X_pool.shape, y_pool,
                                                                win_len=win_len, step=1,
                                                                delta_t=delta_t,
                                                                num_sample=num_sample,
                                                                save_path=os.path.join(save_path,
                                                                                       f"level_{level}_pseudo_labels.png"))
        logger.warning(f"pseudo labels shape: {pseudo_labels.shape}")
        logger.warning(f"subsampled indices shape: {subsampled_indices.shape}")

        sample_indices = subsampled_indices[:, 0]
        counts = np.bincount(sample_indices, minlength=X_pool.shape[0])
        logger.warning(f"Sampling distribution wave-wise: Mean {np.mean(counts):.2f}, Std {np.std(counts):.2f}")

        # Compute window energy
        energies = compute_window_energy(X_pool, subsampled_indices, win_len=win_len)
        logger.warning(f"energies shape: {energies.shape}")

        # Train PixelHop
        model = Model()
        model.fit_saab(X_pool, idx=level, fw=saab_kernel, save_path=save_path)

        # --- Batched On-demand Feature Extraction ---
        logger.warning("Starting batched on-demand SAAB feature extraction...")
        saab = batch_transform_saab_on_demand(
            X_source=X_pool,
            indices_to_process=subsampled_indices,
            model_instance=model,
            level=level,
            win_len=win_len,
            saab_kernel=saab_kernel,
            saab_stride=saab_stride
        )
        logger.warning(f"Batched on-demand feature extraction complete. Final shape: {saab.shape}")

        del X_pool
        gc.collect()

        # Train RFT LNT
        model.train_RFT_LNT(saab, pseudo_labels, idx=level, n_selected=RFT_dims[level], save_path=save_path)
        feat_train = model.Get_RFT_LNT_Features(saab, idx=level)
        logger.warning(f"Feature set {level} RFT LNT: {feat_train.shape}")

        del saab
        gc.collect()

        # Train XGBoost
        all_feat_train = np.concatenate((feat_train, energies), axis=1)

        del feat_train, energies
        gc.collect()

        params = {
            'max_depth': 2,
            'eta': 0.7,
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 10000,
            'n_jobs': 16,
        }
        model.train_xgb(all_feat_train, pseudo_labels, level, params, save_path)

        model.save_model(level, save_path)
        logger.warning(f"level: {level}")
        logger.warning(f"wave_ratio: {wave_ratio}")
        logger.warning(f"num_sample: {num_sample}")
        logger.warning(f"RFT_dim: {RFT_dims[level]}")
        logger.warning(f"SAAB_kernel: {saab_kernel}")
        logger.warning(f"SAAB_stride: {saab_stride}")
        logger.warning(f"XGBoost params {params}")
        logger.warning(f"================================================")


if __name__ == "__main__":
    main()
