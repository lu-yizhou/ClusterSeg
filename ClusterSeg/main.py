import argparse
import logging
import os
import random
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from Dataset import CADataset
from ClusterSeg import *
from configs import get_config
from sklearn.metrics import f1_score
from PIL import Image
from utils import *
from lovasz_losses import *


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--in_channels', type=int, default=3, help='input patch size of network input')
args = parser.parse_args()


def train(args, model):
    global mask_loss, boundary_loss, cluster_loss
    logging.basicConfig(filename='log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size
    T = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                            transforms.RandomOrder([transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomChoice([transforms.RandomRotation((0, 0)),
                                                                             transforms.RandomRotation((90, 90)),
                                                                             transforms.RandomRotation((180, 180)),
                                                                             transforms.RandomRotation((270, 270))])]),
                            transforms.ToTensor()
                            ])
    test_T = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
    trainset = CADataset('dataset/train', transform=T)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = CADataset('dataset/test', transform=test_T)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    mask_loss, boundary_loss, cluster_loss = LovaszSoftmax(), LovaszSoftmax(), LovaszSoftmax()
    for epoch_num in iterator:
        model.train()
        alpha, beta, gamma = 1/3, 1/3, 1/3
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, bound_batch, name = sampled_batch
            image, label, bound = image_batch.cuda(), label_batch.cuda(), bound_batch.cuda()
            mout, bout, cout = model(image)
            boundary_gt = (bound != 0).type(torch.int64).cuda()
            cluster_gt = (bound == 2).type(torch.int64).cuda()
            loss1 = mask_loss(mout[:1], label[:1])
            loss2 = boundary_loss(bout[:1], boundary_gt[:1])
            loss3 = cluster_loss(cout[:1], cluster_gt[:1])
            loss = alpha * loss1 + beta * loss2 + gamma * loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1
            logging.info('iteration %d: loss: %.5f, mask loss: %.5f, boundary loss: %.5f, cluster loss: %.5f' % (iter_num, loss.item(), loss1.item(), loss2.item(), loss3.item()))
        lr_ = base_lr * (1.0 - epoch_num / max_epoch) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        test_loss = []
        IoU_list = []
        F1_score_list = []
        mask_loss_list = []
        bound_loss_list = []
        cluster_loss_list = []
        model.eval()
        for i, batch in enumerate(testloader):
            image, label, bound, name = batch
            image, label, bound = image.cuda(), label.cuda(), bound.cuda()
            boundary_gt = (bound != 0).long().cuda()
            cluster_gt = (bound == 2).long().cuda()
            with torch.no_grad():
                m, b, c = model(image)
                loss1 = mask_loss(m, label)
                loss2 = boundary_loss(b, boundary_gt)
                loss3 = cluster_loss(c, cluster_gt)
                loss = alpha * loss1 + beta * loss2 + gamma * loss3
                mask_loss_list.append(loss1.item())
                bound_loss_list.append(loss2.item())
                cluster_loss_list.append(loss3.item())
                test_loss.append(loss.item())
                m = m.squeeze(0).detach().cpu().numpy()
                prediction = np.argmax(m, 0)
                gt = label.squeeze(0).squeeze(0).detach().cpu().numpy().astype('int')
                IoU = calculate_IoU(prediction, gt)
                F1_score = calculate_F1_score(prediction, gt)
                IoU_list.append(IoU)
                F1_score_list.append(F1_score)
        logging.info('Test: epoch {}: loss: {:.5f}, mask loss: {:.5f}, boundary loss: {:.5f}, cluster loss: {:.5f}, IoU: {:.5f}, F1 score: {:.5f}'
                     .format(epoch_num, np.mean(test_loss), np.mean(mask_loss_list), np.mean(bound_loss_list), np.mean(cluster_loss_list), np.mean(IoU_list), np.mean(F1_score_list)))

    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    torch.save(model.state_dict(), 'ckpt/ClusterSeg.pth')
    logging.info("final model saved")


if __name__ == "__main__":
    config = get_config()
    config.img_size = (args.img_size, args.img_size)
    config.patches.grid = (args.img_size / config.patch_size, args.img_size / config.patch_size)
    net = ClusterSeg(config, img_size=args.img_size, num_classes=args.num_classes, in_channels=args.in_channels).cuda()
    weights = np.load(config.pretrained_path)
    net.load_from(weights=weights)
    train(args, net)
