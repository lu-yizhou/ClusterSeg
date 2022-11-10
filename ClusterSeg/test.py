import argparse
import logging
import os
import random
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import CADataset
from ClusterSeg import *
from configs import get_config
import json


num_classes = 2
batch_size = 1
img_size = 512
seed = 1234
test_save_dir = 'prediction'
logging_file = "log/TransCA25NetTestLog.txt"


def inference(model, test_save_dir=None):
    testset = CADataset('../histology_annotation/dataset/test')
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    if test_save_dir is not None:
        logging.info("prediction saved to {}".format(test_save_dir))
        if not os.path.exists(test_save_dir):
            os.mkdir(test_save_dir)
        if not os.path.exists('prediction_mask_{}'.format(number)):
            os.mkdir('prediction_mask_{}'.format(number))
    else:
        logging.info("prediction results not saved")
    model.eval()
    metric_list = 0.0
    for i_batch, batch in tqdm(enumerate(testloader)):
        image, label, _, name = batch
        image = image.cuda()
        label = label.squeeze(0).squeeze(0).detach().cpu().numpy()
        with torch.no_grad():
            m, b, c = model(image)
            m = m.squeeze(0).detach().cpu().numpy()
            mout = np.argmax(m, 0)
            bout = b.squeeze(0).detach().cpu().numpy()
            bout = np.argmax(bout, 0)
            cout = c.squeeze(0).detach().cpu().numpy()
            cout = np.argmax(cout, 0)
        metric_i = [calculate_acc(mout, label), calculate_IoU(mout, label), calculate_F1_score(mout, label)]
        metric_list += np.array(metric_i)
        if test_save_dir is not None:
            save_path = os.path.join(test_save_dir, '{}.png'.format(name[0]))
            prediction = np.zeros((img_size, img_size, 3), dtype='uint8')
            prediction[mout == 1] = [255, 255, 255]
            prediction[bout == 1] = [0, 255, 0]
            prediction[cout == 1] = [255, 0, 0]
            prediction = Image.fromarray(prediction)
            prediction.save(save_path)
    metric_list = metric_list / len(testset)
    logging.info('Mean accuracy: {:.5f}, mean IoU: {:.5f}, mean F1 score: {:.5f}'.format(metric_list[0], metric_list[1], metric_list[2]))

    print("Testing Finished!")


if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logging.basicConfig(filename=logging_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    config = get_config()
    net = ClusterSeg(config, img_size=img_size, num_classes=num_classes, in_channels=3).cuda()

    model_path = 'ckpt/ClusterSeg.pth'
    net.load_state_dict(torch.load(model_path))

    inference(net, test_save_dir)
