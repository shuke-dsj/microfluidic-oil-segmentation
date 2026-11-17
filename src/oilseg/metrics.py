"""Both image-mean and dataset-global IoU, with absent classes excluded."""
import numpy as np


def confusion_matrix(prediction, target, classes=6):
    if prediction.shape != target.shape:
        raise ValueError('Prediction and target shapes must match')
    if prediction.size == 0 or np.any((prediction < 0) | (prediction >= classes)) or np.any((target < 0) | (target >= classes)):
        raise ValueError('Expected non-empty class-index arrays in range')
    return np.bincount((target.astype(np.int64)*classes + prediction).ravel(), minlength=classes**2).reshape(classes,classes)


def class_iou(matrix):
    inter = np.diag(matrix).astype(float)
    union = matrix.sum(0) + matrix.sum(1) - inter
    return np.divide(inter, union, out=np.full(len(inter),np.nan), where=union>0)
