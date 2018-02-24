import torch
import numpy as np
import cv2
import jpeg4py
from torch.utils.data import Dataset
from pathlib import Path

data_path = Path('data')


class BinaryDataset(Dataset):
    def __init__(self, file_names: list, to_augment=False, transform=None, mode='train'):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        img = load_image(img_file_name)
        mask = load_mask(img_file_name)

        img, mask = self.transform(img, mask)

        if self.mode == 'train':
            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            return to_float_tensor(img), Path(img_file_name).stem


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(path):
    return jpeg4py.JPEG(str(path)).decode()


def load_mask(path):
    mask = cv2.imread(str(path).replace('images', 'binary_masks').replace('jpg', 'png'), 0)
    return (mask > 0).astype(np.uint8)
