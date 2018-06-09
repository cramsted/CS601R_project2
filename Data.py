import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import cv2
from skimage.transform import resize

TRAIN_DATA = "../mscoco_subset_cs601r/train/"
TRAIN_DATA_LABEL = "../mscoco_subset_cs601r/train_masks/"
TEST_DATA = "../mscoco_subset_cs601r/test/"
TEST_DATA_LABEL = "../mscoco_subset_cs601r/test_masks/"


class Data(Dataset):

    def __init__(self, folder_data, folder_labels, transform=None):
        # self.transform = transform

        self._data = sorted(glob.glob(folder_data+"*.jpg"))
        self._labels = sorted(glob.glob(folder_labels+"*.png"))

    def __getitem__(self, index):
        imgData = cv2.cvtColor(cv2.imread(
            self._data[index]), cv2.COLOR_BGR2RGB)
        imgLabel = cv2.cvtColor(cv2.imread(
            self._labels[index]), cv2.COLOR_BGR2GRAY)
        # random flip
        if np.random.rand() > 0.5:
            imgData = np.flip(imgData, axis=1).copy()
            imgLabel = np.flip(imgLabel, axis=1).copy()
        imgData, imgLabel = self.crop(imgData, imgLabel)
        # data
        imgData = np.transpose(imgData, (2, 0, 1))
        display_image = imgData.copy()
        imgData = torch.from_numpy(imgData).float()
        # labels
        # 0 = other
        imgLabel[imgLabel == 90] = 1  # ground
        imgLabel[imgLabel == 120] = 2  # plant
        imgLabel[imgLabel == 150] = 3  # building
        imgLabel[imgLabel == 180] = 4  # sky
        imgLabel[imgLabel == 210] = 5  # vechical
        imgLabel[imgLabel == 240] = 6  # person
        imgLabel = torch.from_numpy(imgLabel).float()
        return imgData, imgLabel, display_image

    def crop(self, img, label):
        min_side = min(img.shape[:-1])
        min_arg = np.argmax(img.shape[:-1])

        # resize an image shape that can be mostly covered by a 224x224 window
        ratio = 300 / min_side
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
        label = cv2.resize(label, (0, 0), fx=ratio, fy=ratio,
                           interpolation=cv2.INTER_NEAREST)
        max_side = max(img.shape[:-1])
        max_arg = np.argmax(img.shape[:-1])
        min_side = 300
        max_start = np.random.randint(0, max_side - 224)
        min_start = np.random.randint(0, min_side - 224)
        if max_arg == 0:
            img = img[max_start:max_start+224, min_start:min_start+224]
            label = label[max_start:max_start+224, min_start:min_start+224]
        else:
            img = img[min_start:min_start+224, max_start:max_start+224]
            label = label[min_start:min_start+224, max_start:max_start+224]
        return img, label

    def __len__(self):
        return len(self._data)


if __name__ == '__main__':
    tr = Data(TRAIN_DATA, TRAIN_DATA_LABEL)
    loader = DataLoader(tr, shuffle=True)
    data, label = next(iter(loader))
    import pdb
    pdb.set_trace()
