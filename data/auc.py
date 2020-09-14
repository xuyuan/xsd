import pandas as pd
from sklearn.metrics import roc_curve, auc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute AUC')
    parser.add_argument('gt_csv', default=None, metavar='GT_CSV',
                        type=str, help="Path to the ground truch csv file")
    parser.add_argument('pred_csv', default=None, metavar='PRED_PATH',
                        type=str, help="Path to the predicted csv file")

    args = parser.parse_args()

    labels = pd.read_csv(args.gt_csv, na_filter=False)
    gt = labels.annotation.astype(bool).astype(float).values

    pred_csv = pd.read_csv(args.pred_csv, na_filter=False)
    pred = pred_csv.prediction.values

    assert pred_csv.image_path.equals(labels.image_name)

    fpr, tpr, _ = roc_curve(gt, pred)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)
