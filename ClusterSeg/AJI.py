import numpy as np
import cv2
import os


def AJI(gt, output):
    n_ins = gt.max()
    n_out = output.max()
    if n_out == 0:
        if n_ins == 0:
            return 1
        else:
            return 0
    empty = 0
    Iand = 0
    Ior = 0
    for i in range(n_out):
        out_table = np.where(output == i + 1, 1, 0)
        max_and = 0
        max_or = 0
        for j in range(n_ins):
            gt_table = np.where(gt == j + 1, 1, 0)
            ior1 = np.sum(out_table + gt_table > 0)
            iand1 = np.sum(out_table) + np.sum(gt_table) - ior1
            if (iand1 > max_and):
                max_and = iand1
                max_or = ior1
        if max_and == 0:
            empty = empty + np.sum(out_table)
        Iand += max_and
        Ior += max_or
    return Iand / (Ior + empty)



