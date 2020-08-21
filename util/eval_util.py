import os.path
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from . import anom_utils


def eval_ood_measure(conf, pred, seg_label, mask=None):
    correct_map = pred == seg_label
    out_label = np.logical_not(correct_map)

    in_scores = - conf[np.logical_not(out_label)]
    out_scores  = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any OOD pixels or is only OOD.")
        return None

def eval_alarm_metrics(pred_ious, real_ious):
    mae = np.nanmean(np.abs(np.array(pred_ious) - np.array(real_ious)), axis=0)
    std = np.nanstd(np.abs(np.array(pred_ious) - np.array(real_ious)), axis=0)
    classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                            'bicycle']
    pred_ious = np.array(pred_ious)
    real_ious = np.array(real_ious)
    pcs = []
    scs = []
    for cls, cls_name in enumerate(classes):
        valid_inds = np.logical_not(np.isnan(real_ious[:,cls]))
        pc, p = stats.pearsonr(real_ious[:,cls][valid_inds], pred_ious[:,cls][valid_inds])
        sc, p = stats.spearmanr(real_ious[:,cls][valid_inds], pred_ious[:,cls][valid_inds])
        #print('%s, correlation coefficient: %.3f, p value: %.6f' % (cls_name, r, p))
        pcs.append(pc)
        scs.append(sc)
    print(("{},"*len(classes)).format(*classes))
    print(("P.C. = "+"{:.6f},"*len(classes)).format(*pcs))
    print("mean P.C. = ", np.nanmean(pcs))
    print(("S.C. = "+"{:.6f},"*len(classes)).format(*scs))
    print("mean S.C. = ", np.nanmean(scs))
    print(("MAE = "+"{:.6f},"*len(classes)).format(*mae))
    print("mmae = ", np.nanmean(mae))
    print(("STD = "+"{:.6f},"*len(classes)).format(*std))
    print("mstd = ", np.nanmean(std))

    print('copy paste')
    print(("{:.6f},"*len(classes)).format(*pcs))
    print(("{:.6f},"*len(classes)).format(*scs))
    print(("{:.6f},"*len(classes)).format(*mae))
    print(("{:.6f},"*len(classes)).format(*std))



def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def result_stats(hist):
    acc_overall = np.diag(hist).sum() / hist.sum() * 100
    acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) * 100
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    freq = hist.sum(1) / hist.sum()
    fwIU = (freq[freq > 0] * iu[freq > 0]).sum()
    pix_percls = hist.sum(1)
    return acc_overall, acc_percls, iu, fwIU, pix_percls


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist


class Metrics:
    def __init__(self, metrics, len_dataset, n_classes):
        self.metrics = metrics
        self.len_dataset = len_dataset
        self.n_classes = n_classes
        self.accurate, self.errors, self.proba_pred = [], [], []
        self.accuracy = 0
        self.current_miou = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, pred, target, confidence):
        self.accurate.extend(pred.eq(target.view_as(pred)).detach().to("cpu").numpy())
        self.accuracy += pred.eq(target.view_as(pred)).sum().item()
        self.errors.extend((pred != target.view_as(pred)).detach().to("cpu").numpy())
        self.proba_pred.extend(confidence.detach().to("cpu").numpy())

        if "mean_iou" in self.metrics:
            pred = pred.cpu().numpy().flatten()
            target = target.cpu().numpy().flatten()
            mask = (target >= 0) & (target < self.n_classes)
            hist = np.bincount(
                self.n_classes * target[mask].astype(int) + pred[mask],
                minlength=self.n_classes ** 2,
            ).reshape(self.n_classes, self.n_classes)
            self.confusion_matrix += hist

    def get_scores(self, split="test"):
        self.accurate = np.reshape(self.accurate, newshape=(len(self.accurate), -1)).flatten()
        self.errors = np.reshape(self.errors, newshape=(len(self.errors), -1)).flatten()
        self.proba_pred = np.reshape(self.proba_pred, newshape=(len(self.proba_pred), -1)).flatten()

        scores = {}
        if "accuracy" in self.metrics:
            accuracy = self.accuracy / self.len_dataset
            scores[f"{split}/accuracy"] = {"value": accuracy, "string": f"{accuracy:05.2%}"}
        if "auc" in self.metrics:
            if len(np.unique(self.accurate)) == 1:
                auc = 1
            else:
                auc = roc_auc_score(self.accurate, self.proba_pred)
            scores[f"{split}/auc"] = {"value": auc, "string": f"{auc:05.2%}"}
        if "ap_success" in self.metrics:
            ap_success = average_precision_score(self.accurate, self.proba_pred)
            scores[f"{split}/ap_success"] = {"value": ap_success, "string": f"{ap_success:05.2%}"}
        if "accuracy_success" in self.metrics:
            accuracy_success = np.round(self.proba_pred[self.accurate == 1]).mean()
            scores[f"{split}/accuracy_success"] = {
                "value": accuracy_success,
                "string": f"{accuracy_success:05.2%}",
            }
        if "ap_errors" in self.metrics:
            ap_errors = average_precision_score(self.errors, -self.proba_pred)
            scores[f"{split}/ap_errors"] = {"value": ap_errors, "string": f"{ap_errors:05.2%}"}
        if "accuracy_errors" in self.metrics:
            accuracy_errors = 1.0 - np.round(self.proba_pred[self.errors == 1]).mean()
            scores[f"{split}/accuracy_errors"] = {
                "value": accuracy_errors,
                "string": f"{accuracy_errors:05.2%}",
            }
        if "fpr_at_95tpr" in self.metrics:
            for i,delta in enumerate(np.arange(
                self.proba_pred.min(),
                self.proba_pred.max(),
                (self.proba_pred.max() - self.proba_pred.min()) / 1000,
            )):
                tpr = len(self.proba_pred[(self.accurate == 1) & (self.proba_pred >= delta)]) / len(
                    self.proba_pred[(self.accurate == 1)]
                )
                if i%100 == 0:
                    print(f"Threshold:\t {delta:.6f}")
                    print(f"TPR: \t\t {tpr:.4%}")
                    print("------")
                if 0.9505 >= tpr >= 0.9495:
                    print(f"Nearest threshold 95% TPR value: {tpr:.6f}")
                    print(f"Threshold 95% TPR value: {delta:.6f}")
                    fpr = len(
                        self.proba_pred[(self.errors == 1) & (self.proba_pred >= delta)]
                    ) / len(self.proba_pred[(self.errors == 1)])
                    scores[f"{split}/fpr_at_95tpr"] = {"value": fpr, "string": f"{fpr:05.2%}"}
                    break
        if "mean_iou" in self.metrics:
            iou = np.diag(self.confusion_matrix) / (
                self.confusion_matrix.sum(axis=1)
                + self.confusion_matrix.sum(axis=0)
                - np.diag(self.confusion_matrix)
            )
            mean_iou = np.nanmean(iou)
            scores[f"{split}/mean_iou"] = {"value": mean_iou, "string": f"{mean_iou:05.2%}"}

        return scores