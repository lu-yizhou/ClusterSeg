import os
import random
import h5py
import numpy as np
from PIL import Image
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, label):
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        return image, label.long()


class CADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.label_dir = os.path.join(self.root_dir, 'label')
        self.bound_dir = os.path.join(self.root_dir, 'bound')
        self.images = os.listdir(self.data_dir)
        self.labels = os.listdir(self.label_dir)
        self.bounds = os.listdir(self.bound_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        base_name = os.path.splitext(name)[0]
        image_path = os.path.join(self.data_dir, name)
        label_path = os.path.join(self.label_dir, name)
        bound_path = os.path.join(self.bound_dir, name)
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        bound = Image.open(bound_path).convert('L')

        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            image = self.transform(image)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            label = self.transform(label).long()

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            bound = self.transform(bound)
            bound = (bound * 255).long()
        else:
            T = transforms.Compose([
                transforms.ToTensor()
            ])
            image = T(image)
            label = T(label).long()
            bound = T(bound)
            bound = (bound * 255).long()
        return image, label.squeeze(0), bound.squeeze(0), base_name
