from utils import check_mkdir
import numpy as np
import os
from model import Model
from loguru import logger
from joblib import Parallel, delayed
import time
import warnings
import argparse
import sys
from math import ceil
from collections import defaultdict
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.callback import TrainingCallback
from utils import plot_xgb_learning_curve
from tqdm.auto import tqdm


class LogCallback(TrainingCallback):
    """XGBoost callback that logs train/val metrics each iteration via loguru."""

    def after_iteration(self, model, epoch, evals_log):
        # build a line like "[1]   train-rmse:0.12345   val-rmse:0.23456"
        log_message = f"[{epoch + 1}]"
        for data_name, metrics in evals_log.items():
            for metric_name, history in metrics.items():
                # history is a list of values; take the latest
                log_message += f"   {data_name}-{metric_name}:{history[-1]:.5f}"
        logger.warning(log_message)
        # return False means “do not stop training”
        return False


def process(idxs, y_train, model_path, pid):
    from loguru import logger
    logger.remove()
    formatter = "{time:YYYY-MM-DD HH:mm:ss} | {message}"
    logger.add(sys.stdout, level="WARNING", format=formatter)

    p_model = Model()
    p_model.load_model(os.path.join(model_path, 'p_wave'))

    s_model = Model()
    s_model.load_model(os.path.join(model_path, 's_wave'))

    batch_results = []
    for idx in tqdm(idxs, total=len(idxs), desc=f"Process {pid}", position=pid, leave=False, dynamic_ncols=True):
        X = X_train[idx]
        res = defaultdict(list)
        res['y'] = y_train[idx]

        # P-wave
        p_preds, p_traces = p_model.predict(X, win_len_list, active_levels, p_peak_dists, saab_stride_list)
        for lvl in active_levels:
            pred = int(p_preds[lvl])
            trace = p_traces[lvl]
            res['feat'].append(trace[pred])
            res['feat'].append(pred)
        p_pred_final = p_preds[top_level] * (2 ** top_level)

        # S-wave
        s_preds, s_traces = s_model.predict(X, win_len_list, active_levels, s_peak_dists, saab_stride_list,
                                            min_location=p_pred_final)
        for lvl in active_levels:
            pred = int(s_preds[lvl])
            trace = s_traces[lvl]
            res['feat'].append(trace[pred])
            res['feat'].append(pred)

        res['feat'].append(p_preds[top_level] - s_preds[top_level])
        res['feat'] = np.array(res['feat'])
        batch_results.append(res)

    return batch_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--wave_ratio', type=float, default=0.01)

    args = parser.parse_args()
    exp_name = args.exp_name
    n_jobs = args.n_jobs
    wave_ratio = args.wave_ratio

    warnings.filterwarnings("ignore")
    start_time = time.time()
    ckpt_path = './exp'
    save_path = os.path.join(ckpt_path, exp_name, f'detection')
    data_path = './data/'
    check_mkdir(save_path)

    # Set up logging
    log_path = os.path.join(save_path, 'logfile.log')
    logger.add(log_path, enqueue=True, level="WARNING", format="{time:YYYY-MM-DD HH:mm:ss} | {message}")

    # Load data
    X_train = np.load(os.path.join(data_path, "X_train.npy"), mmap_mode='r')
    y_train = np.load(os.path.join(data_path, "P_train.npy")).astype(np.float32)
    y_train = (y_train >= 0).astype(np.int32)

    N_waves = X_train.shape[0]
    wave_size = int(N_waves * wave_ratio)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(N_waves, size=wave_size, replace=False))
    X_train = X_train[idx]
    y_train = y_train[idx]
    logger.warning(f"Wave ratio: {wave_ratio} from {N_waves}, wave_size: {wave_size}")

    # Start prediction
    SAAB_BATCH_SIZE = 1_000

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

    total_samples = X_train.shape[0]
    model_path = os.path.join(ckpt_path, exp_name)

    logger.warning(f"Processing {total_samples} samples with {n_jobs} CPUs")
    all_idxs = list(range(total_samples))
    chunk_size = ceil(total_samples / n_jobs)
    chunks = [all_idxs[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process)(chunk, y_train, model_path, pid)
        for pid, chunk in enumerate(chunks)
    )
    results = [item for batch in results for item in batch]
    feats = np.stack([r['feat'] for r in results], axis=0)  # (N, D)
    y = np.array([r['y'] for r in results], dtype=int)  # (N,)

    indices = np.arange(feats.shape[0])
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=0)
    X_train, X_val = feats[train_indices], feats[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    logger.warning(f'Train XGBoost, train {X_train.shape}, val {X_val.shape}')

    callbacks = [LogCallback(), xgb.callback.EarlyStopping(rounds=5, save_best=True)]

    params = {
        'max_depth': 2,
        'eta': 0.3,
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 10000,
        'n_jobs': 16,
    }

    evals_result = {}
    clf = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        callbacks=callbacks,
        evals_result=evals_result,
        verbose_eval=False
    )
    n_trees = clf.num_boosted_rounds()
    logger.warning(f'Number of trees: {n_trees}')
    plot_xgb_learning_curve(evals_result, eval_metric="logloss", path=os.path.join(save_path, f"detection_logloss.png"))

    clf.save_model(os.path.join(save_path, f'detection.model'))
