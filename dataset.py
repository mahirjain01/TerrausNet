import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import prepare_data
from albumentations.pytorch.transforms import ToTensorV2

import glob
import os

import scipy.io
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Grayscale, RandomCrop


class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return ToTensor()(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return ToTensor()(image), torch.from_numpy(mask).long()
        else:
            return ToTensor()(image), str(img_file_name)


class HistologyDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', problem_type=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.file_names = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = load_image(img_path)
        mask = load_mask(mask_path, self.problem_type)

        data = {"image": image, "mask": mask}
        # print("The data is : ", data)
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return ToTensor()(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return ToTensor()(image), torch.from_numpy(mask).long()
        else:
            return ToTensor()(image), str(img_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    mask = cv2.imread(str(path))
    return mask.astype(np.uint8)


# histology = HistologyDataset('/home/arnav/Disk/HistologyNet/robot-surgery-segmentation/data/HistologyNet/labelled')
# print(histology[0])
