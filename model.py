import os
import numpy as np
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from FeatureTest import FeatureTest
from LNT import LNT
from PixelHop import Pixelhop
from utils import plot_engery, plot_train_val_rank, Shrink, Concat, plot_xgb_learning_curve, compute_window_energy, \
    average_pooling, build_center_indices, post_process_prob, plot_rft_lnt_features, extract_saab_windows
import pickle
from loguru import logger
from xgboost.callback import TrainingCallback
import re
import gc


class LogCallback(TrainingCallback):
    """XGBoost callback that logs train/val metrics each iteration via loguru."""

    def after_iteration(self, model, epoch, evals_log):
        # build a line like "[1]   train-rmse:0.12345   val-rmse:0.23456"
        log_message = f"[{epoch + 1}]"
        for data_name, metrics in evals_log.items():
            for metric_name, history in metrics.items():
                # history is a list of values; take the latest
                log_message += f"   {data_name}-{metric_name}:{history[-1]:.5f}"
        logger.info(log_message)
        # return False means “do not stop training”
        return False


class Model:
    def __init__(self):
        self.pixelhop_lst = {}
        self.rft_lst = {}
        self.lnt_lst = {}
        self.xgb_lst = {}

    def save_model(self, idx, save_path):
        res = {
            'pixelhop': self.pixelhop_lst[idx],
            'rft': self.rft_lst[idx],
            'lnt': self.lnt_lst[idx],
        }
        with open(os.path.join(save_path, f'model_{idx}.pkl'), 'wb') as f:
            pickle.dump(res, f)
        self.xgb_lst[idx].save_model(os.path.join(save_path, f'xgboost_{idx}.model'))
        logger.info(f"Saved model to {save_path}")

    def load_model(self, save_path):
        pattern = re.compile(r"model_(\d+)\.pkl")
        for filename in os.listdir(save_path):
            match = pattern.match(filename)
            if match:
                idx = int(match.group(1))
                model_path = os.path.join(save_path, filename)
                with open(model_path, 'rb') as f:
                    res = pickle.load(f)
                self.pixelhop_lst[idx] = res['pixelhop']
                self.rft_lst[idx] = res['rft']
                self.lnt_lst[idx] = res['lnt']

        pattern = re.compile(r"xgboost_(\d+)\.model")
        for filename in os.listdir(save_path):
            match = pattern.match(filename)
            if match:
                idx = int(match.group(1))
                model_path = os.path.join(save_path, filename)
                reg = xgb.Booster()
                reg.load_model(model_path)
                self.xgb_lst[idx] = reg

    # @profile
    def fit_saab(self, X, idx, fw, save_path):
        SaabArgs = [{'num_AC_kernels': -1, 'needBias': False, 'cw': False}]
        shrinkArgs = [{'func': Shrink, 'filter_width': fw, 'stride': 1, 'padding': fw - 1}]
        concatArg = {'func': Concat}
        p = Pixelhop(depth=1, TH1=0, TH2=0, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)

        rng = np.random.default_rng(seed=42)
        if X.shape[0] > 150_000:
            train_indices = rng.choice(X.shape[0], size=150_000, replace=False)
            X = X[train_indices]
        logger.info(f"Training PixelHop {idx}, {X.shape}")
        p.fit(X)

        del X
        gc.collect()

        plot_engery(p, title=f"PixelHop{idx} Energy Plot",
                    path=os.path.join(save_path, f"level_{idx}_energy.png"))
        self.pixelhop_lst[idx] = p
        return p

    # @profile
    def transform_saab(self, X, idx, save_path=None):
        p = self.pixelhop_lst[idx]
        batches = np.array_split(X, np.ceil(X.shape[0] / 1_000))

        first_saab = p.transform(X[:1])[0]
        res_shape = (X.shape[0],) + first_saab.shape[1:]

        if save_path is not None:
            temp_file_path = os.path.join(save_path, f'saab_{idx}.mmap')
            res = np.memmap(temp_file_path, dtype=np.float32, mode='w+', shape=res_shape)
        else:
            res = np.empty(res_shape, dtype=np.float32)

        current_pos = 0
        for batch in batches:
            res[current_pos:current_pos + len(batch)] = p.transform(batch)[0]
            current_pos += len(batch)

        logger.info(f"Get PixelHop {idx}, {res.shape}")
        return res

    # @profile
    def train_RFT_LNT(self, X, Y, idx, n_selected, save_path):
        logger.info(f"Training RFT & LNT for feature set {idx}")
        X = X.reshape(X.shape[0], -1)
        sample_size = min(1_000_000, Y.shape[0])
        X_sample, Y_sample = X[:sample_size], Y[:sample_size]

        X_train, X_val, y_train, y_val = train_test_split(X_sample, Y_sample, test_size=0.2, random_state=0)
        logger.info(f'Train RFT, shape: {X_train.shape}')
        rft = FeatureTest('rmse')
        rft.fit(X_train, y_train, n_bins=16, outliers=True)
        rft.plot(path=os.path.join(save_path, f"level_{idx}_train_rft.png"))

        logger.info(f'Val RFT, shape: {X_val.shape}')
        rft_val = FeatureTest('rmse')
        rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
        rft_val.plot(path=os.path.join(save_path, f"level_{idx}_val_rft.png"))

        plot_train_val_rank(rft, rft_val, path=os.path.join(save_path, f"level_{idx}_joint.png"))

        # Train and validation overlap
        overlap_feat = np.intersect1d(rft.sorted_features[:n_selected], rft_val.sorted_features[:n_selected])
        rft.sorted_features = overlap_feat
        rft.n_selected = len(overlap_feat)

        X_train = rft.transform(X_train, n_selected=rft.n_selected)
        X_val = rft.transform(X_val, n_selected=rft.n_selected)
        self.rft_lst[idx] = rft

        lnt = LNT(no_tree=300, max_depth=3, lr=0.1, feature_in_comb=50)
        lnt.fit(X_train, y_train, X_val, y_val)
        logger.info(f"Get LNT {lnt.dim} features")
        self.lnt_lst[idx] = lnt

        X_train = np.concatenate((X_train, lnt.transform(X_train)), axis=-1)

        plot_rft_lnt_features(X_train, y_train, rft, lnt, idx, save_path)

    def Get_RFT_LNT_Features(self, X, idx):
        X = X.reshape(X.shape[0], -1)
        rft = self.rft_lst[idx]
        res = rft.transform(X, n_selected=rft.n_selected)
        logger.info(f"Get {rft.n_selected} RFT features from {rft.dim}")

        lnt = self.lnt_lst[idx]
        lnt_feat = lnt.transform(res)
        res = np.concatenate((res, lnt_feat), axis=-1).astype(np.float32)
        return res

    # @profile
    def train_xgb(self, X, Y, idx, params, save_path):
        indices = np.arange(X.shape[0])
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=0)
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = Y[train_indices], Y[val_indices]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        logger.info(f'Train XGBoost, train {X_train.shape}, val {X_val.shape}')

        callbacks = [LogCallback(), xgb.callback.EarlyStopping(rounds=5, save_best=True)]

        evals_result = {}
        reg = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=callbacks,
            evals_result=evals_result,
            verbose_eval=False
        )
        self.xgb_lst[idx] = reg
        n_trees = reg.num_boosted_rounds()
        logger.warning(f'Number of trees: {n_trees}')

        plot_xgb_learning_curve(evals_result, eval_metric="rmse", path=os.path.join(save_path, f"level_{idx}_rmse.png"))

        # Train score report
        y_pred_train = reg.predict(dtrain)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        logger.info(f'mse_train: {mse_train} mae_train: {mae_train}')

        # Validation score report
        y_pred_val = reg.predict(dval)
        mse_val = mean_squared_error(y_val, y_pred_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        logger.info(f"mse_val: {mse_val} mae_val: {mae_val}")

        y_final = np.zeros_like(Y)
        y_final[train_indices] = y_pred_train
        y_final[val_indices] = y_pred_val
        return y_final

    def predict(self, X, win_len_list, active_levels, peak_dists, saab_stride, min_location=None, roi_width=20,
                thr_mul=0.95):
        level_preds = {}
        level_probs_trace = {}
        X = X[np.newaxis, :, :]
        N, C, L_orig = X.shape
        roi_bounds = None  # (N,2) ROI [left,right] for each sample, updated layer by layer

        for i, level in enumerate(active_levels):
            win_len = win_len_list[level]

            # 1. Average pooling to current level
            if level == 0:
                X_pool = X
            else:
                X_pool = average_pooling(X, num_transform=level)
            N, C, L = X_pool.shape

            # 2. Generate center indices (full scan in coarsest layer, scan ROI in other layers)
            min_location_level = None
            if min_location is not None:
                downsample_factor = 2 ** level
                min_location_level = int(min_location / downsample_factor)
            indices = build_center_indices(1, L, win_len, step=1, roi_bounds=roi_bounds)
            if indices is None:
                for lvl in active_levels[i:]:
                    level_preds[lvl] = -1
                    level_probs_trace[lvl] = np.zeros(L_orig // (2 ** (lvl)), dtype=np.float32)
                return level_preds, level_probs_trace
            centers = indices[:, 1]

            # 3. Energy + PixelHop + Other features
            energies = compute_window_energy(X_pool, indices, win_len)
            saab = self.transform_saab(X_pool, level)
            saab_win = extract_saab_windows(saab, indices, win_len, stride=saab_stride[level])

            feat = self.Get_RFT_LNT_Features(saab_win, level)
            all_feat = np.concatenate((feat, energies), axis=-1)

            # 4. XGBoost predict
            raw_pred = self.xgb_lst[level].predict(xgb.DMatrix(all_feat))
            # raw_pred = np.clip(raw_pred, 0, 1)

            # 5. Generate probability curve and post-process
            prob_trace = np.zeros(L, dtype=np.float32)
            prob_trace[centers] = raw_pred

            if min_location_level is not None:
                prob_trace[:min_location_level] = 0

            pick = post_process_prob(prob_trace, peak_dist=peak_dists[level], thr_mul=thr_mul)
            if pick is None:
                pick = int(np.argmax(prob_trace))
            level_preds[level] = pick
            level_probs_trace[level] = prob_trace

            # 6. Calculate next level ROI
            if i < len(active_levels) - 1:
                next_level = active_levels[i + 1]
                scale = 2 ** (level - next_level)
                left = max(0, (pick - roi_width) * scale)
                right = min(L_orig - 1, (pick + roi_width) * scale)
                roi_bounds = np.array([[left, right]], dtype=int)

        return level_preds, level_probs_trace
