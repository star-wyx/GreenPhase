from sklearn.linear_model import LinearRegression
import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm._classes")


class LNT:
    def __init__(self, no_tree, max_depth, lr, feature_in_comb):
        self.A = None
        self.no_tree = no_tree
        self.max_depth = max_depth
        self.lr = lr
        self.feature_in_comb = feature_in_comb

    def fit(self, X_train, y_train, X_val, y_val):
        logger.info(f"Training LNT")

        if X_train.shape[0] > 80000:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_train.shape[0], size=80000, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        if X_val.shape[0] > 20000:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_val.shape[0], size=20000, replace=False)
            X_val = X_val[idx]
            y_val = y_val[idx]

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': self.max_depth,
            'learning_rate': self.lr
        }

        watchlist = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(params, dtrain,
                          num_boost_round=self.no_tree,
                          evals=watchlist, early_stopping_rounds=10,
                          verbose_eval=False, evals_result={})

        best_iteration = model.best_iteration
        logger.info(f"Best iteration: {best_iteration}")

        tree_paths = self.get_path(model)

        num_features = X_train.shape[1]
        A = np.zeros((num_features, len(tree_paths)))

        logger.info(f'# of combination {len(tree_paths)}')
        for i in range(len(tree_paths)):
            selected = tree_paths[i]
            selected = [int(idx) for idx in selected]
            X_sel = X_train[:, selected]

            lin_reg = LinearRegression()
            lin_reg.fit(X_sel, y_train)
            theta = np.zeros(num_features)
            theta[selected] = lin_reg.coef_
            A[:, i] = theta

        self.A = A
        self.dim = A.shape[1]

    def transform(self, X):
        if 'scaler' in self.__dict__:
            X = self.scaler.transform(X)
        res_lnt = (X @ self.A).astype(np.float32)
        return res_lnt

    def get_path(self, model):
        trees_df = model.trees_to_dataframe()
        tree_ids = trees_df['Tree'].unique()
        tree_feat = []

        for tree_id in tree_ids:
            tree_df = trees_df[trees_df['Tree'] == tree_id]
            tmp = set()
            for _, row in tree_df.iterrows():
                feature = row['Feature']
                if feature != 'Leaf':
                    tmp.add(int(feature[1:]))
            tree_feat.append(sorted(tmp))

        res = []
        tmp = set()
        for i in range(len(tree_feat)):
            if len(tmp) < self.feature_in_comb:
                tmp = tmp.union(tree_feat[i])
            else:
                res.append(list(tmp))
                tmp = set()
        if len(tmp) > 0:
            res.append(list(tmp))

        return res
