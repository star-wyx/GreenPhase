from utils import check_mkdir, map_label_after_pooling
import numpy as np
import os
from model import Model
from loguru import logger
from joblib import Parallel, delayed
import time
import warnings
import argparse
from math import ceil
import xgboost as xgb
import pandas as pd
from tqdm import tqdm


def process(idxs, p_true_all, s_true_all, model_path, pid):
    # 1) set up logger once
    from loguru import logger
    logger.remove()

    # 2) load both models once
    p_model = Model()
    p_model.load_model(os.path.join(model_path, 'p_wave'))

    s_model = Model()
    s_model.load_model(os.path.join(model_path, 's_wave'))

    det_clf = xgb.Booster()
    det_clf.load_model(os.path.join(model_path, 'detection', 'detection.model'))

    batch_results = []
    for idx in tqdm(idxs, total=len(idxs), desc=f"worker {pid}", position=pid, leave=False, dynamic_ncols=True):
        X = X_test[idx]
        p_true = p_true_all[idx]
        s_true = s_true_all[idx]
        event_id = event_ID_test[idx]
        det_feat = []
        event_row = {'Event_id': event_id}

        # P-wave
        p_preds, p_traces = p_model.predict(X, win_len_list, active_levels, p_peak_dists, saab_stride_list)
        p_truths = {lvl: (map_label_after_pooling(p_true, num_transform=lvl) if lvl != 0 else p_true) for lvl in
                    active_levels}
        for lvl in active_levels:
            pred = int(p_preds[lvl])
            trace = p_traces[lvl]
            gt = int(p_truths[lvl])
            prob_at_pred = trace[pred]
            event_row[f'p_{lvl}_prediction'] = pred
            event_row[f'p_{lvl}_gt'] = gt
            event_row[f'p_{lvl}_prob'] = f"{prob_at_pred:.4f}"
            det_feat.append(trace[pred])
            det_feat.append(pred)
        # Convert to the original resolution
        p_pred_final = p_preds[top_level] * (2 ** top_level)

        # S-wave
        s_preds, s_traces = s_model.predict(X, win_len_list, active_levels, s_peak_dists, saab_stride_list,
                                            min_location=p_pred_final)
        s_truths = {lvl: (map_label_after_pooling(s_true, num_transform=lvl) if lvl != 0 else s_true) for lvl in
                    active_levels}
        for lvl in active_levels:
            pred = int(s_preds[lvl])
            trace = s_traces[lvl]
            gt = int(s_truths[lvl])
            prob_at_pred = trace[pred]
            event_row[f's_{lvl}_prediction'] = pred
            event_row[f's_{lvl}_gt'] = gt
            event_row[f's_{lvl}_prob'] = f"{prob_at_pred:.4f}"
            det_feat.append(trace[pred])
            det_feat.append(pred)
        # Convert to the original resolution
        s_pred_final = s_preds[top_level] * (2 ** top_level)

        # Detection task
        det_feat.append(p_preds[top_level] - s_preds[top_level])
        det_feat = np.array(det_feat, dtype=np.float32).reshape(1, -1)
        det_pred = det_clf.predict(xgb.DMatrix(det_feat))
        event_row['detection'] = f"{det_pred[0]:.4f}"
        event_row['p_final_prediction'] = p_pred_final
        event_row['p_gt'] = p_true
        event_row['s_final_prediction'] = s_pred_final
        event_row['s_gt'] = s_true
        batch_results.append(event_row)

    return batch_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=16)

    args = parser.parse_args()
    exp_name = args.exp_name
    n_jobs = args.n_jobs

    warnings.filterwarnings("ignore")
    start_time = time.time()
    ckpt_path = './exp'
    save_path = os.path.join(ckpt_path, exp_name, f'test')
    data_path = './data/'
    check_mkdir(save_path)

    # Set up logging
    log_path = os.path.join(save_path, 'logfile.log')
    logger.add(log_path, enqueue=True, level="WARNING", format="{time:YYYY-MM-DD HH:mm:ss} | {message}")

    # Load data
    X_test = np.load(os.path.join(data_path, "X_test.npy"), mmap_mode='r')
    p_true = np.load(os.path.join(data_path, "P_test.npy"))
    s_true = np.load(os.path.join(data_path, "S_test.npy"))
    event_ID_test = np.load(os.path.join(data_path, "Event_test.npy"))
    ori_samples = X_test.shape[0]
    logger.warning(f"X_test shape: {X_test.shape}")

    # Config
    win_len_list = {
        4: 33,  # 375
        3: 65,  # 750
        2: 129,  # 1500
        1: 257,  # 3000
        0: 513,  # 6000
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
    active_levels = sorted(saab_kernel_list.keys(), reverse=True)
    top_level = active_levels[-1]
    roi_width = 20
    p_peak_dists = {lvl: max(1, int(400 / (2 ** lvl))) for lvl in active_levels}
    s_peak_dists = {lvl: max(1, int(4000 / (2 ** lvl))) for lvl in active_levels}

    total_samples = X_test.shape[0]
    model_path = os.path.join(ckpt_path, exp_name)

    # Start prediction
    logger.warning(f"Processing {total_samples} samples with {n_jobs} CPUs")
    all_idxs = list(range(total_samples))
    chunk_size = ceil(total_samples / n_jobs)
    chunks = [all_idxs[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process)(chunk, p_true, s_true, model_path, pid)
        for pid, chunk in enumerate(chunks)
    )
    results = [item for batch in results for item in batch]

    # Write results to csv
    logger.warning("Writing results to a csv")
    header = ['Event_id', 'detection']
    for lvl in active_levels:
        header.extend([f'p_{lvl}_prediction', f'p_{lvl}_gt', f'p_{lvl}_prob'])
        header.extend([f's_{lvl}_prediction', f's_{lvl}_gt', f's_{lvl}_prob'])
    header.extend(['p_final_prediction', 'p_gt', 's_final_prediction', 's_gt'])

    results_df = pd.DataFrame(results)
    results_df = results_df[header]
    output_csv_path = os.path.join(save_path, 'results.csv')
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    logger.warning(f"Results have been saved to: {output_csv_path}")

    end = time.time()
    elapsed = (end - start_time) / 3600
    est_total_h = (elapsed / len(X_test)) * ori_samples
    logger.warning(f"Total time: {elapsed:.2f}h, Estimated full run: {est_total_h:.2f}h")
    logger.warning(f"active_levels: {active_levels}")
    logger.warning(f"roi_width: {roi_width}")
