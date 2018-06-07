import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
from PIL import Image
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
        # data
        imgData = cv2.cvtColor(cv2.imread(
            self._data[index]), cv2.COLOR_BGR2RGB)
        imgData = cv2.resize(imgData, (224, 224))
        imgData = np.array(
            [imgData[:, :, 0, ], imgData[:, :, 1], imgData[:, :, 2]])
        imgData = torch.from_numpy(imgData).float()
        # labels
        imgLabel = cv2.cvtColor(cv2.imread(
            self._labels[index]), cv2.COLOR_BGR2GRAY)
        imgLabel = cv2.resize(imgLabel, (224, 224),
                              interpolation=cv2.INTER_NEAREST)
        # 0 = other
        imgLabel[imgLabel == 90] = 1  # ground
        imgLabel[imgLabel == 120] = 2  # plant
        imgLabel[imgLabel == 150] = 3  # building
        imgLabel[imgLabel == 180] = 4  # sky
        imgLabel[imgLabel == 210] = 5  # vechical
        imgLabel[imgLabel == 240] = 6  # person
        imgLabel = torch.from_numpy(imgLabel).float()
        return imgData, imgLabel
    # def __getitem__(self, index):
    #     imgData = Image.open(self._data[index]).convert("RGB")
    #     # imgData = resize(imgData, (224, 224))
    #     imgLabel = Image.open(self._labels[index]).convert("RGB")
    #     # imgLabel = resize(imgLabel, (224, 224))

    #     imgData = torch.from_numpy(resize(np.asarray(imgData), (224, 224)))
    #     imgLabel = torch.from_numpy(resize(np.asarray(imgLabel), (224, 224)))
    #     # import pdb
    #     # pdb.set_trace()
    #     return imgData, imgLabel

    def __len__(self):
        return len(self._data)


if __name__ == '__main__':
    tr = Data(TRAIN_DATA, TRAIN_DATA_LABEL)
    loader = DataLoader(tr, shuffle=True)
    data, label = next(iter(loader))
    import pdb
    pdb.set_trace()
