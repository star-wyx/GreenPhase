import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
from loguru import logger


def evaluate_picking(det_pred, preds, trues, roi=10, phase='p'):
    actual = trues >= 0
    correct = np.zeros_like(actual, dtype=bool)
    idxs = np.where(actual & det_pred)[0]
    correct[idxs] = np.abs(preds[idxs] - trues[idxs]) <= roi

    tp = int((actual & det_pred & correct).sum())
    fp = int((~actual & det_pred).sum() +
             (actual & det_pred & ~correct).sum())
    fn = int((actual & ~det_pred).sum())
    pos_total = int(actual.sum())

    acc_pos = tp / pos_total if pos_total > 0 else 0.0
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    logger.warning(f"{phase} Picking: Acc_pos={acc_pos:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
    return acc_pos, prec, rec, f1


def evaluate_detection(det_pred, trues):
    actual = trues >= 0
    acc = accuracy_score(actual, det_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(actual, det_pred, average='binary')
    logger.warning(f"Detection: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
    return acc, prec, rec, f1


def load_csv(csv_path, phase):
    df = pd.read_csv(csv_path)
    detection_col = 'detection'
    pred_col = f'{phase}_final_prediction'
    true_col = f'{phase}_gt'

    detection_probs = df[detection_col].values
    preds = df[pred_col].values
    trues = df[true_col].values

    return detection_probs, preds, trues


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()
    exp_name = args.exp_name

    save_path = os.path.join('./exp', exp_name, 'test')
    log_path = os.path.join(save_path, 'logfile.log')
    logger.add(log_path, enqueue=True, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {message}")

    csv = f'./exp/{exp_name}/test/results.csv'
    det_probs, p_preds, p_trues = load_csv(csv, 'p')
    _, s_preds, s_trues = load_csv(csv, 's')

    thre = 0.5
    mask = det_probs > thre
    _, _, _, det_f1 = evaluate_detection(mask, p_trues)
    _, _, _, p_f1 = evaluate_picking(mask, p_preds, p_trues, roi=50, phase='P')
    _, _, _, s_f1 = evaluate_picking(mask, s_preds, s_trues, roi=50, phase='S')
