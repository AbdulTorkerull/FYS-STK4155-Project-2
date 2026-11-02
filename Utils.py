import numpy as np

def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    cm = np.zeros((2, 2), dtype=int)
    for y, y_hat in zip(y_true, y_pred):
        cm[1 - y, 1 - y_hat] += 1

    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    return tp, fn, fp, tn


def classification_report(y_true, y_pred):
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred)
    n = len(y_true.ravel())

    accuracy = (tp + tn)/n
    precision = tp/(tp + fp)
    recall = tp/(fp + fn)
    F1 = (2*precision*recall)/(precision + recall)

    return accuracy, precision, recall, F1

def roc_curve(y_true, scores):
    y_true = np.asarray(y_true).ravel().astype(int)
    scores = np.asarray(scores).ravel().astype(float)

    order = np.argsort(-scores)
    y, scores = y_true[order], scores[order]

    thresholds = np.sort(np.unique(scores))[::-1]
    thresholds = np.insert(thresholds, 0, np.inf)

    TPR = np.zeros(len(thresholds))
    FPR = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        y_hat = (scores >= thresholds[i]).astype(int)
        tp, fn, fp, tn = confusion_matrix(y, y_hat)
        TPR[i] = tp/(tp + fn)
        FPR[i] = fp/(fp + tn)

    return FPR, TPR, thresholds

def auc_trapezoid(fpr, tpr):
    lengths = np.diff(fpr)
    heights = (tpr[1:] + tpr[:-1])/2
    auc = np.sum(lengths*heights)
    
    return auc

def roc_auc(y_true, scores):
    y_true = np.asarray(y_true).ravel().astype(int)
    scores = np.asarray(scores).ravel().astype(float)

    order = np.argsort(scores)
    ranks = np.argsort(order) + 1.0
    for uniques in np.unique(scores):
        clones = (scores == uniques)
        ranks[clones] = ranks[clones].mean()

    R = ranks[y_true == 1]
    n_pos = len(scores[y_true == 1])
    n_neg = len(scores[y_true == 0])

    sum_R = R.sum()
    AUC = (sum_R - (1/2)*n_pos*(n_pos + 1))/(n_pos*n_neg)

    return AUC

def cross_validation():
    pass