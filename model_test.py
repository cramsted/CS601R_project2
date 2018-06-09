import torch.nn as nn
import torchvision.models as models
from torchvision.utils import make_grid
from Data import Data
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import pickle
import numpy as np
from SegNet import SegNet
import matplotlib.pyplot as plt
import cv2

TRAIN_DATA = "../mscoco_subset_cs601r/train/"
TRAIN_DATA_LABEL = "../mscoco_subset_cs601r/train_masks/"
TEST_DATA = "../mscoco_subset_cs601r/test/"
TEST_DATA_LABEL = "../mscoco_subset_cs601r/test_masks/"

MODEL_FILENAME = 'model.json'
# MODEL_FILENAME = 'model2.json'


def show(img):
    npimg = img.cpu().detach().numpy()
    # import pdb
    # pdb.set_trace()
    output = npimg[0, :, :]
    for i in range(1, 7):
        output = np.hstack((output, npimg[i, :, :]))
    plt.imshow(output, interpolation='nearest')


def convertImage(img):
    img = img.cpu().detach().numpy()
    img = np.array([img[0, :, :], img[1, :, :], img[2, :, :]])
    img = np.transpose(img, (1, 2, 0))
    # import pdb
    # pdb.set_trace()
    # return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def convertLabel(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = flatten_batch(img)
    img = np.array([img, img, img])
    img = np.transpose(img, (1, 2, 0))
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = convertLabel2RGB(img)
    # import pdb
    # pdb.set_trace()
    # img.astype(np.float64)
    return img


def convertLabel2RGB(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
        # 0 = other
    img[img == 1] = 90  # ground
    img[img == 2] = 120  # plant
    img[img == 3] = 150  # building
    img[img == 4] = 180  # sky
    img[img == 5] = 210  # vechical
    img[img == 6] = 240  # person
    img = img.astype(np.float64)
    return img


def convertPrediction(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = flatten_batch(img)
    img = np.array([img, img, img])
    img = np.transpose(img, (1, 2, 0))
    return convertLabel2RGB(img)


def flatten_batch(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()

    output = img[0]
    for i in range(1, img.shape[0]):
        output = np.hstack((output, img[i]))
    return output


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
seg = SegNet()
seg.to(device)
try:
    seg.load_state_dict(torch.load(MODEL_FILENAME))
except:
    pass

dataset = Data(TEST_DATA, TEST_DATA_LABEL)
loader = DataLoader(dataset, batch_size=5, shuffle=False)

with torch.no_grad():
    for data in loader:
        images, labels = data[0].to(device), data[1].to(device)
        images = images.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.LongTensor).cuda()

        prediction = seg(images)
        prediction_mask = torch.argmax(prediction, 1)

        # plt.title(MODEL_FILENAME)
        # plt.subplot(311)
        # plt.title("Images")
        # plt.imshow(convertImage(make_grid(images, padding=5)))
        # plt.subplot(312)
        # plt.title("Labels")
        # plt.imshow(convertLabel(labels))
        # plt.subplot(313)
        # plt.title("Predictions")
        # plt.imshow(convertPrediction(prediction_mask))
        # plt.show()
        # break
