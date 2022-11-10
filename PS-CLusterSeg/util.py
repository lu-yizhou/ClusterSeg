import os
import numpy as np
import torch
from PIL import Image
from ClusterSeg import ClusterSeg
from sklearn.metrics import f1_score

# import networks

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


def build_model(config):
    net = ClusterSeg(config, img_size=config.img_size[0], num_classes=config.n_classes, in_channels=3)
    weights = np.load(config.pretrained_path)
    net.load_from(weights=weights)
    return net


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
    IoU = (np.sum(intersection) + 1e-6) / (np.sum(union) + 1e-6)
    return IoU


def calculate_F1_score(prediction, label):
    y_true = label.reshape(-1)
    y_pred = prediction.reshape(-1)
    return f1_score(y_true, y_pred)


def calculate_dice(prediction, label):
    # equal to F1 score
    intersection = np.logical_and(prediction, label)
    dice = 2 * np.sum(intersection) / (np.sum(prediction) + np.sum(label))
    return dice
