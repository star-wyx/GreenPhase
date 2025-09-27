import os
import re
import pickle
import xgboost as xgb
from model import Model

SCALE_LIST = {
    1: 4,  # 6000 // 1500
    2: 8,  # 6000 // 750
    3: 16,  # 6000 // 375
}

WIN_LEN_LIST = {
    1: 129,
    2: 65,
    3: 33, 
}

DELTA_T_LIST = {
    1: 12,
    2: 6,
    3: 3,
}

SAAB_KERNEL_LIST = {
    1: 16,
    2: 8,
    3: 8,
}
SAAB_STRIDE_LIST = {
    1: 8,
    2: 8,
    3: 4
}

RFT_DIMS = {
    1: 200,
    2: 120,
    3: 80,
}


def load_model(save_path):
    pixelhop_lst = {}
    rft_lst = {}
    lnt_lst = {}
    xgb_lst = {}

    pattern = re.compile(r"model_(\d+)\.pkl")
    for filename in os.listdir(save_path):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            model_path = os.path.join(save_path, filename)
            with open(model_path, 'rb') as f:
                res = pickle.load(f)
            pixelhop_lst[idx] = res['pixelhop']
            rft_lst[idx] = res['rft']
            lnt_lst[idx] = res['lnt']

    pattern = re.compile(r"xgboost_(\d+)\.model")
    for filename in os.listdir(save_path):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            model_path = os.path.join(save_path, filename)
            reg = xgb.Booster()
            reg.load_model(model_path)
            xgb_lst[idx] = reg

    return pixelhop_lst, rft_lst, lnt_lst, xgb_lst


if __name__ == "__main__":

    for wave in ['p', 's']:
        save_path = f"./exp/demo/{wave}_wave/"

        pixelhop_lst, rft_lst, lnt_lst, xgb_lst = load_model(save_path)
        model = Model(SCALE_LIST, WIN_LEN_LIST, SAAB_STRIDE_LIST)
        model.pixelhop_lst = pixelhop_lst
        model.rft_lst = rft_lst
        model.lnt_lst = lnt_lst
        model.xgb_lst = xgb_lst
        model.scale_list = SCALE_LIST

        model.save_model(1, save_path)
        model.save_model(2, save_path)
        model.save_model(3, save_path)