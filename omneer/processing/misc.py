import numpy as np
from collections.abc import Sequence
from scipy import interp
import pandas as pd
from typing import List, Tuple, Dict
from collections.abc import Iterable
from sklearn.metrics import log_loss, brier_score_loss
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    cohen_kappa_score,
    precision_recall_fscore_support,
    classification_report,
    multilabel_confusion_matrix,
    hamming_loss,
    jaccard_score,
)


_depth = lambda L: isinstance(L, (Sequence, np.ndarray)) and max(map(_depth, L)) + 1


def handle_input(value):
    return value if isinstance(value, Iterable) else [value]

def compute_metric(metric_func, y_true, y_pred, **kwargs):
    try:
        return metric_func(y_true, y_pred, **kwargs)
    except Exception as e:
        print(f"Error computing {metric_func.__name__}: {str(e)}")
        return np.nan


def compute_ci(metric_values, ci_level=0.95):
    """
    Compute confidence interval for a given metric.
    
    :param metric_values: Array-like of metric values.
    :param ci_level: Confidence level (default: 0.95).
    :return: Tuple of lower and upper bounds of the confidence interval.
    """
    n = len(metric_values)
    mean = np.mean(metric_values)
    se = stats.sem(metric_values)
    margin = se * stats.t.ppf((1 + ci_level) / 2, n - 1)
    lower = mean - margin
    upper = mean + margin
    return lower, upper


def get_metrics(y_true_all: list, y_preds_all: list, scores_all: list, ci_level=0.95) -> pd.DataFrame:
    """
    Compute multiple classification metrics for predictions and scores.

    :param y_true_all: List of true labels.
    :param y_preds_all: List of predicted labels.
    :param scores_all: List of prediction scores.
    :param ci_level: Confidence level for computing confidence intervals (default: 0.95).
    :return: DataFrame with the mean, standard deviation, and confidence intervals of each metric.
    """
    metrics = []
    for y_true, y_pred, scores in zip(y_true_all, y_preds_all, scores_all):
        metrics_record = defaultdict()

        metrics_record['Confusion Matrix'] = confusion_matrix(y_true, y_pred)
        metrics_record['Accuracy'] = compute_metric(accuracy_score, y_true, y_pred)
        metrics_record['Precision'] = compute_metric(precision_score, y_true, y_pred, average='weighted')
        metrics_record['Recall'] = compute_metric(recall_score, y_true, y_pred, average='weighted')
        metrics_record['F1 Score'] = compute_metric(f1_score, y_true, y_pred, average='weighted')
        metrics_record['Matthews Corr.'] = compute_metric(matthews_corrcoef, y_true, y_pred)
        metrics_record['AUC (ROC)'] = compute_metric(roc_auc_score, y_true, scores, average='weighted')
        metrics_record['AUC (PR)'] = compute_metric(average_precision_score, y_true, scores, average='weighted')
        metrics_record['Log Loss'] = compute_metric(log_loss, y_true, scores)
        metrics_record['Brier Score Loss'] = compute_metric(brier_score_loss, y_true, scores)
        metrics_record['Cohen\'s Kappa'] = compute_metric(cohen_kappa_score, y_true, y_pred)

        # Additional metrics for multi-class classification
        if len(np.unique(y_true)) > 2:
            metrics_record['Hamming Loss'] = compute_metric(hamming_loss, y_true, y_pred)
            metrics_record['Jaccard Score'] = compute_metric(jaccard_score, y_true, y_pred, average='weighted')

            precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
            class_report = classification_report(y_true, y_pred, output_dict=True)
            for class_label, class_metrics in class_report.items():
                if class_label != 'accuracy':
                    metrics_record[f"Precision (Class {class_label})"] = class_metrics['precision']
                    metrics_record[f"Recall (Class {class_label})"] = class_metrics['recall']
                    metrics_record[f"F1 Score (Class {class_label})"] = class_metrics['f1-score']
                    metrics_record[f"Support (Class {class_label})"] = class_metrics['support']

        metrics.append(metrics_record)

    met = pd.DataFrame(metrics)
    met_summary = met.loc[:, met.columns != 'Confusion Matrix'].agg(['mean', 'std'])
    
    # Compute confidence intervals
    ci_summary = met.loc[:, met.columns != 'Confusion Matrix'].apply(compute_ci, ci_level=ci_level)
    
    return met_summary


def pr_interp(rc_, rc, pr):

    pr_ = np.zeros_like(rc_)
    locs = np.searchsorted(rc, rc_)

    for idx, loc in enumerate(locs):
        l = loc - 1
        r = loc
        r1 = rc[l] if l > -1 else 0
        r2 = rc[r] if r < len(rc) else 1
        p1 = pr[l] if l > -1 else 1
        p2 = pr[r] if r < len(rc) else 0

        t1 = (1 - p2) * r2 / p2 / (r2 - r1) if p2 * (r2 - r1) > 1e-16 else (1 - p2) * r2 / 1e-16
        t2 = (1 - p1) * r1 / p1 / (r2 - r1) if p1 * (r2 - r1) > 1e-16 else (1 - p1) * r1 / 1e-16
        t3 = (1 - p1) * r1 / p1 if p1 > 1e-16 else (1 - p1) * r1 / 1e-16

        a = 1 + t1 - t2
        b = t3 - t1 * r1 + t2 * r1
        pr_[idx] = rc_[idx] / (a * rc_[idx] + b)

    return pr_


def get_roc_info(y_true_all, scores_all):

    fpr_pt = np.linspace(0, 1, 1001)
    tprs, aucs = [], []

    for i in range(len(y_true_all)):
        y_true = y_true_all[i]
        scores = scores_all[i]
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=scores, drop_intermediate=True)
        tprs.append(interp(fpr_pt, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))

    tprs_mean = np.mean(tprs, axis=0)
    tprs_std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tprs_mean + tprs_std, 1)
    tprs_lower = np.maximum(tprs_mean - tprs_std, 0)
    auc_mean = auc(fpr_pt, tprs_mean)
    auc_std = np.std(aucs)

    rslt = {
        'xs': fpr_pt,
        'ys_mean': tprs_mean,
        'ys_upper': tprs_upper,
        'ys_lower': tprs_lower,
        'auc_mean': auc_mean,
        'auc_std': auc_std
    }

    return rslt


def get_pr_info(y_true_all, scores_all):

    rc_pt = np.linspace(0, 1, 1001)
    rc_pt[0] = 1e-16
    prs = []
    aps = []

    for i in range(len(y_true_all)):
        y_true = y_true_all[i]
        scores = scores_all[i]
        pr, rc, _ = precision_recall_curve(y_true=y_true, probas_pred=scores)
        aps.append(average_precision_score(y_true=y_true, y_score=scores))
        pr, rc = pr[::-1], rc[::-1]
        prs.append(pr_interp(rc_pt, rc, pr))

    prs_mean = np.mean(prs, axis=0)
    prs_std = np.std(prs, axis=0)
    prs_upper = np.minimum(prs_mean + prs_std, 1)
    prs_lower = np.maximum(prs_mean - prs_std, 0)
    aps_mean = np.mean(aps)
    aps_std = np.std(aps)

    rslt = {
        'xs': rc_pt,
        'ys_mean': prs_mean,
        'ys_upper': prs_upper,
        'ys_lower': prs_lower,
        'auc_mean': aps_mean,
        'auc_std': aps_std
    }

    return rslt


def l1_regularizer(model, lambda_l1=0.01):
    ''' LASSO '''

    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

def elastic_net_regularizer(model, lambda_l1=0.01, lambda_l2=0.01):
    ''' ElasticNet Regularization '''

    loss_en = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            loss_en += ElasticNet(alpha=lambda_l1, l1_ratio=lambda_l2).fit(model_param_value.abs().sum()).coef_
    return loss_en

def get_calibration_curve(y_true_all: List, y_preds_all: List) -> Dict[str, np.array]:
    """
    Computes calibration curve.

    :param y_true_all: List of true labels.
    :param y_preds_all: List of predicted labels.
    :return: Fraction of positives and mean predicted values.
    """
    frac_of_pos, mean_pred_value = calibration_curve(y_true_all, y_preds_all, n_bins=10)

    return {
        'Fraction of positives': frac_of_pos,
        'Mean predicted value': mean_pred_value
    }

def pr_auc_score(y_true: np.array, y_preds: np.array) -> float:
    """
    Compute area under the precision-recall curve. More robust than average_precision_score.

    :param y_true: True labels.
    :param y_preds: Predicted labels.
    :return: Area Under the Precision-Recall Curve (AUPRC).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_preds)
    return auc(recall, precision)