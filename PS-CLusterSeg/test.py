import os
import sys
from tqdm import tqdm
import argparse
import logging
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from configs import get_config
import ramps
from dataset import CADataset
from PIL import Image
from ClusterSeg import *


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='histology_dataset', help='Name of Experiment')
parser.add_argument('--exp', type=int,  default=1, help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=3, help='labeled_batch_size per gpu')
parser.add_argument('--plabeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--unlabeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float,  default=0.9, help='learning rate decay')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=7.0, help='consistency_rampup')
parser.add_argument('--scale', type=int,  default=512, help='batch size of 8 with resolution of 416*416 is exactly OK')
args = parser.parse_args()

test_data_path = os.path.join(args.root_path, 'test')
snapshot_path = "./exp_{}_batch_size_{}_labeled_bs_{}_plabeled_bs_{}_unlabeled_bs_{}_base_lr_{}_consistency_type_{}/".format(args.exp, args.batch_size, args.labeled_bs, args.plabeled_bs, args.unlabeled_bs, args.base_lr, args.consistency_type)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
lr_decay = args.lr_decay
loss_record = 0
num_classes = 2


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    test_save_dir = os.path.join(snapshot_path, 'prediction')
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)
    if not os.path.exists(os.path.join(snapshot_path, 'prediction_mask')):
        os.mkdir(os.path.join(snapshot_path, 'prediction_mask'))

    config = get_config()
    model = ClusterSeg(config, img_size=args.scale, num_classes=num_classes, in_channels=3).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_{}.pth'.format(args.max_iterations))
    state_dict = torch.load(save_mode_path)
    model.load_state_dict(state_dict)

    db_test = CADataset(root=test_data_path)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    model.eval()

    for i_batch, sampled_batch in enumerate(testloader):
        with torch.no_grad():
            image_batch, _, _, name = sampled_batch['image'], sampled_batch['label'], sampled_batch['bound'], sampled_batch['name']
            image_batch = image_batch.cuda()
            mout, bout, cout = model(image_batch)

            mout = mout.squeeze(0).detach().cpu().numpy()
            mout = np.argmax(mout, 0)
            bout = bout.squeeze(0).detach().cpu().numpy()
            bout = np.argmax(bout, 0)
            cout = cout.squeeze(0).detach().cpu().numpy()
            cout = np.argmax(cout, 0)

            save_path = os.path.join(test_save_dir, '{}.png'.format(name[0]))
            prediction = np.zeros((args.scale, args.scale, 3), dtype='uint8')
            prediction[mout == 1] = [255, 255, 255]
            prediction[bout == 1] = [0, 255, 0]
            prediction[cout == 1] = [255, 0, 0]
            prediction = Image.fromarray(prediction)
            prediction.save(save_path)

