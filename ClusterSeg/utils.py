import numpy as np
import torch
from sklearn.metrics import f1_score


def calculate_acc(prediction, label):
    h, w = label.shape
    x, y = prediction.shape
    assert h == x and w == y
    total = h * w
    correct = np.sum(prediction == label)
    return correct / total


def calculate_IoU(prediction, label):
    h, w = label.shape
    x, y = prediction.shape
    assert h == x and w == y
    intersection = np.logical_and(prediction, label)
    union = np.logical_or(prediction, label)
    if np.sum(union) == 0:
        assert np.sum(intersection) == 0
        IoU = 1
    else:
        IoU = np.sum(intersection) / np.sum(union)
    return IoU


def calculate_F1_score(prediction, label):
    intersection = np.logical_and(prediction, label)
    dice = 2 * np.sum(intersection) / (np.sum(prediction) + np.sum(label))
    return dice
