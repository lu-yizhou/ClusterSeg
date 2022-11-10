import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from configs import get_config
import ramps
from dataset import CADataset
from util import *
import losses


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./histology_dataset', help='Name of Experiment')
parser.add_argument('--exp', type=int,  default=1, help='number')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--plabeled_bs', type=int, default=0, help='labeled_batch_size per gpu')
parser.add_argument('--unlabeled_bs', type=int, default=0, help='labeled_batch_size per gpu')
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


train_data_path = os.path.join(args.root_path, 'train')
snapshot_path = "./exp_{}_batch_size_{}_labeled_bs_{}_plabeled_bs_{}_unlabeled_bs_{}_base_lr_{}_consistency_type_{}/".format(args.exp, args.batch_size, args.labeled_bs, args.plabeled_bs, args.unlabeled_bs, args.base_lr, args.consistency_type)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
batch_size = args.batch_size * len(args.gpu.split(','))

max_iterations = args.max_iterations
base_lr = args.base_lr
lr_decay = args.lr_decay
loss_record = 0
num_classes = 2


def get_current_consistency_weight(epoch, max_epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, max_epoch)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(ema=False):
    # Network definition
    config = get_config()
    net = build_model(config)
    net_cuda = net.cuda()
    if ema:
        for param in net_cuda.parameters():
            param.detach_()
    return net_cuda


def add_noise(image, lam=0.3):
    # image: [B, C, H, W], 0-1
    RGB = torch.exp(image) / torch.sum(torch.exp(image), dim=1, keepdim=True).cuda()
    B, C, H, W = image.shape
    mean = torch.zeros((B, C, H, W)).cuda()
    std = torch.std(image.view(B, C, H * W), dim=2, keepdim=True).expand(B, C, H * W).view(B, C, H, W)
    noise = torch.normal(mean, std) * lam * RGB
    noise = torch.clamp(noise, -0.2, 0.2)
    ema_image = torch.clamp(image + image * noise, 0, 1)  # thyroid
    return ema_image


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=os.path.join(snapshot_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = create_model()
    ema_model = create_model(ema=True)

    T = transforms.Compose([transforms.Resize((args.scale, args.scale)),
                            transforms.RandomOrder([transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomChoice([transforms.RandomRotation((0, 0)),
                                                                             transforms.RandomRotation((90, 90)),
                                                                             transforms.RandomRotation((180, 180)),
                                                                             transforms.RandomRotation((270, 270))])]),
                            transforms.ToTensor()
                            ])

    db_train = CADataset(root=train_data_path, transform=None)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False)
    model.train()
    ema_model.train()
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * base_lr},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': base_lr, 'weight_decay': 0.0005}
    ], momentum=0.9)

    if args.consistency_type == 'sig_mse':
        consistency_criterion = losses.sigmoid_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = F.kl_div
    elif args.consistency_type == 'mse':
        consistency_criterion = F.mse_loss
    else:
        assert False, args.consistency_type
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        mask_loss_record, mask_con_loss_record, boundary_loss_record, boundary_con_loss_record, cluster_loss_record, cluster_con_loss_record = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            optimizer.param_groups[0]['lr'] = 2 * base_lr * (1 - float(iter_num) / max_iterations) ** lr_decay
            optimizer.param_groups[1]['lr'] = base_lr * (1 - float(iter_num) / max_iterations) ** lr_decay
            image_batch, label_batch, bound_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['bound']
            image_batch, label_batch, bound_batch = image_batch.cuda(), label_batch.cuda(), bound_batch.cuda()
            ema_inputs = add_noise(image_batch)
            mout, bout, cout = model(image_batch)
            with torch.no_grad():
                ema_mout, ema_bout, ema_cout = ema_model(ema_inputs)
            MaskLoss = losses.LovaszSoftmax()
            BoundaryLoss = losses.LovaszSoftmax()
            ClusterLoss = losses.LovaszSoftmax()
            alpha, beta, gamma = 1/3, 1/3, 1/3

            labeled_bs = int(args.labeled_bs)
            plabeled_bs = int(args.plabeled_bs)
            unlabeled_bs = int(args.unlabeled_bs)

            mask_loss = MaskLoss(mout[:labeled_bs], label_batch[:labeled_bs])
            mask_con_loss = consistency_criterion(mout[labeled_bs:(labeled_bs + plabeled_bs + unlabeled_bs)], ema_mout[labeled_bs:(labeled_bs + plabeled_bs + unlabeled_bs)])

            boundary_gt = (bound_batch != 0).cuda()
            boundary_loss = BoundaryLoss(bout[:labeled_bs], boundary_gt[:labeled_bs])
            boundary_con_loss = consistency_criterion(bout[labeled_bs:(labeled_bs + plabeled_bs + unlabeled_bs)], ema_bout[labeled_bs:(labeled_bs + plabeled_bs + unlabeled_bs)])

            cluster_gt = (bound_batch == 2).detach().cpu().numpy().astype('int')
            cluster_gt = torch.tensor(cluster_gt).cuda()
            cluster_loss = ClusterLoss(cout[:(labeled_bs + plabeled_bs)], cluster_gt[:(labeled_bs + plabeled_bs)])
            cluster_con_loss = consistency_criterion(cout[(labeled_bs+plabeled_bs):(labeled_bs+plabeled_bs+unlabeled_bs)], ema_cout[(labeled_bs+plabeled_bs):(labeled_bs+plabeled_bs+unlabeled_bs)])

            supervised_loss = alpha * mask_loss + beta * boundary_loss + gamma * cluster_loss
            consistency_loss = alpha * mask_con_loss + beta * boundary_con_loss + gamma * cluster_con_loss
            consistency_weight = get_current_consistency_weight(epoch_num, max_epoch)

            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1

            mask_loss_record.update(mask_loss.item(), labeled_bs)
            boundary_loss_record.update(boundary_loss.item(), labeled_bs)
            cluster_loss_record.update(cluster_loss.item(), labeled_bs)
            mask_con_loss_record.update(mask_con_loss.item(), batch_size-labeled_bs)
            boundary_con_loss_record.update(boundary_con_loss.item(), batch_size-labeled_bs)
            cluster_con_loss_record.update(cluster_con_loss.item(), batch_size-labeled_bs)

            logging.info('iteration %d : mask loss: %.5f, boundary loss: %.5f, cluster loss: %.5f, mask con loss: %.5f, boundary con loss: %.5f, cluster con loss: %.5f, loss_weight: %.5f, lr: %.5f' %
                         (iter_num, mask_loss_record.avg, boundary_loss_record.avg, cluster_loss_record.avg, mask_con_loss_record.avg, boundary_con_loss_record.avg, cluster_con_loss_record.avg, consistency_weight, optimizer.param_groups[1]['lr']))
            loss_record = 'iteration %d : mask loss: %.5f, boundary loss: %.5f, cluster loss: %.5f, mask con loss: %.5f, boundary con loss: %.5f, cluster con loss: %.5f, loss_weight: %.5f, lr: %.5f' % \
                         (iter_num, mask_loss_record.avg, boundary_loss_record.avg, cluster_loss_record.avg, mask_con_loss_record.avg, boundary_con_loss_record.avg, cluster_con_loss_record.avg, consistency_weight, optimizer.param_groups[1]['lr'])

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    # save_mode_path_ema = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '_ema.pth')
    torch.save(model.state_dict(), save_mode_path)
    # torch.save(ema_model.state_dict(), save_mode_path_ema)
    logging.info("save model to {}".format(save_mode_path))
    with open(os.path.join(snapshot_path, 'loss_record_MTMT.txt'), 'a') as f:
        f.write(str(loss_record)+'\r\n')
