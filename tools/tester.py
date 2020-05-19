import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def calc_eer(targets: np.ndarray, scores: np.ndarray):
    """
    Calculates Equal error rate score
    https://stackoverflow.com/questions/28339746/equal-error-rate-in-python

    Parameters:
    targets: np.ndarray - binary labels in range {0, 1}
    scores: np.ndarray - predicted probabilities for the spoof class

    Returns:
    eer: float - equal error rate score
    threshold: float - threshold boundary
    """
    
    fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, thresholds)(eer)

    return eer * 100, float(threshold)
