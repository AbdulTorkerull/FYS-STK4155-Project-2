"""File containing other utilty functions that are relevant to the project"""

import numpy as np

def one_hot(y, n_classes=10):
    """Function for turning a vector of integer class labels into a one hot encoded matrix with shape (samples) x (classes)"""

    y = np.asarray(y, dtype=int) # Make sure labels are an integer array
    Y = np.zeros((y.size, n_classes)) # Matrix of zeros, one row per sample and one column per class
    Y[np.arange(y.size), y] = 1.0 # Put a 1 in the column of the correct class for each sample

    return Y

def confusion_matrix(y_true, y_pred):
    """Function for computing the confusion matrix of a binary classification problem. Returns the number of true positives (tp),
    false negatives (fn), false positives (fp) and true negatives (tn)"""

    y_true = np.asarray(y_true).ravel().astype(int) # Flatten true labels into a vector of integers
    y_pred = np.asarray(y_pred).ravel().astype(int) # Flatten predicted labels into a vector of integers

    cm = np.zeros((2, 2), dtype=int) # Confusion matrix with the positive class (1) in the first row and column
    for y, y_hat in zip(y_true, y_pred): # Iterate over true and predicted label of each sample
        cm[1 - y, 1 - y_hat] += 1 # 1 - label maps class 1 to index 0 and class 0 to index 1

    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1] # Pick out the four entries of the confusion matrix

    return tp, fn, fp, tn


def classification_report(y_true, y_pred):
    """Function for computing accuracy, precision, recall and F1 score of a binary classification problem"""

    tp, fn, fp, tn = confusion_matrix(y_true, y_pred) # Confusion matrix entries
    n = len(y_true.ravel()) # Number of samples

    accuracy = (tp + tn)/n # Fraction of samples classified correctly
    precision = tp/(tp + fp) # Fraction of predicted positives that are actually positive
    recall = tp/(fp + fn) # Recall as written in the code
    F1 = (2*precision*recall)/(precision + recall) # Harmonic mean of precision and recall

    return accuracy, precision, recall, F1

def roc_curve(y_true, scores):
    """Function for computing the ROC curve of a binary classifier from its scores (for example predicted probabilities).
    Returns the false positive rate, true positive rate and the thresholds used"""

    y_true = np.asarray(y_true).ravel().astype(int) # Flatten true labels into a vector of integers
    scores = np.asarray(scores).ravel().astype(float) # Flatten scores into a vector of floats

    order = np.argsort(-scores) # Indices that sort the scores from highest to lowest
    y, scores = y_true[order], scores[order] # Sort labels and scores in the same order

    thresholds = np.sort(np.unique(scores))[::-1] # Every unique score used as a threshold, from highest to lowest
    thresholds = np.insert(thresholds, 0, np.inf) # Add infinity as first threshold so the curve starts at (0, 0)

    TPR = np.zeros(len(thresholds)) # Array for storing the true positive rate of each threshold
    FPR = np.zeros(len(thresholds)) # Array for storing the false positive rate of each threshold
    for i in range(len(thresholds)): # Iterate over thresholds
        y_hat = (scores >= thresholds[i]).astype(int) # Classify as positive if the score is at or above the threshold
        tp, fn, fp, tn = confusion_matrix(y, y_hat) # Confusion matrix entries at this threshold
        TPR[i] = tp/(tp + fn) # True positive rate, fraction of actual positives found
        FPR[i] = fp/(fp + tn) # False positive rate, fraction of actual negatives wrongly classified as positive

    return FPR, TPR, thresholds

def auc_trapezoid(fpr, tpr):
    """Function for computing the area under the ROC curve using the trapezoidal rule"""

    lengths = np.diff(fpr) # Width of each trapezoid along the FPR axis
    heights = (tpr[1:] + tpr[:-1])/2 # Average of the two TPR values at the edges of each trapezoid
    auc = np.sum(lengths*heights) # Sum the areas of all the trapezoids
    
    return auc

def roc_auc(y_true, scores):
    """Function for computing the area under the ROC curve using ranks (the Mann-Whitney U statistic), 
    which gives the same result as integrating the ROC curve without having to compute it"""

    y_true = np.asarray(y_true).ravel().astype(int) # Flatten true labels into a vector of integers
    scores = np.asarray(scores).ravel().astype(float) # Flatten scores into a vector of floats

    order = np.argsort(scores) # Indices that sort the scores from lowest to highest
    ranks = np.argsort(order) + 1.0 # Rank of each score, starting at 1 for the lowest
    for uniques in np.unique(scores): # Iterate over unique scores to handle ties
        clones = (scores == uniques) # Mask of all samples sharing this score
        ranks[clones] = ranks[clones].mean() # Give tied samples the average of their ranks

    R = ranks[y_true == 1] # Ranks of the positive samples
    n_pos = len(scores[y_true == 1]) # Number of positive samples
    n_neg = len(scores[y_true == 0]) # Number of negative samples

    sum_R = R.sum() # Sum of the ranks of the positive samples
    AUC = (sum_R - (1/2)*n_pos*(n_pos + 1))/(n_pos*n_neg) # Rank sum formula for AUC, normalized by the number of positive/negative pairs

    return AUC