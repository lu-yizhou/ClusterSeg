import os
import os.path
import numpy as np
import random
import torch.utils.data as data
from PIL import Image
import torch
from util import cal_subitizing
from torchvision import transforms
import json


class CADataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.data = os.listdir(os.path.join(self.root, 'data'))
        self.label = os.listdir(os.path.join(self.root, 'label'))
        self.bound = os.listdir(os.path.join(self.root, 'bound'))
        self.transform = transform

    def __getitem__(self, index):
        name = self.data[index]
        base_name = os.path.splitext(name)[0]
        img_path = os.path.join(self.root, 'data', name)
        gt_path = os.path.join(self.root, 'label', name)
        bound_path = os.path.join(self.root, 'bound', name)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        bound = Image.open(bound_path).convert('L')

        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img = self.transform(img)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            target = self.transform(target)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            bound = self.transform(bound)
            bound = (bound * 255).type(torch.int64)
        else:
            T = transforms.Compose([
                transforms.ToTensor()
            ])
            img = T(img)
            target = T(target)
            bound = T(bound)
            bound = (bound * 255).type(torch.int64)
        sample = {'image': img, 'label': target, 'bound': bound, 'name': base_name}
        return sample

    def __len__(self):
        return len(self.data)
