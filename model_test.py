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
from sklearn.metrics import confusion_matrix

TRAIN_DATA = "../mscoco_subset_cs601r/train/"
TRAIN_DATA_LABEL = "../mscoco_subset_cs601r/train_masks/"
TEST_DATA = "../mscoco_subset_cs601r/test/"
TEST_DATA_LABEL = "../mscoco_subset_cs601r/test_masks/"

# MODEL_FILENAME = 'model.json'
# MODEL_FILENAME = 'model2.json'
# MODEL_FILENAME = 'model3.json'
MODEL_FILENAME = 'model4.json'


def show(img):
    npimg = img.cpu().detach().numpy()
    output = npimg[0, :, :]
    for i in range(1, 7):
        output = np.hstack((output, npimg[i, :, :]))
    plt.imshow(output, interpolation='nearest')


def convertImage(img):
    img = img.cpu().detach().numpy()
    img = np.array([img[0, :, :], img[1, :, :], img[2, :, :]])
    img = np.transpose(img, (1, 2, 0))
    return img


def convertLabel(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = flatten_batch(img)
    img = convertLabel2RGB(img)
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
    return convertLabel2RGB(img)


def flatten_batch(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()

    output = img[0]
    for i in range(1, img.shape[0]):
        output = np.hstack((output, img[i]))
    return output


def getIOU(img, label):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy()
    output = {i: {"iou": []} for i in range(7)}
    for j in range(img.shape[0]):
        for i in range(7):
            ground_truth = label == i
            predicted_mask = img == i
            # ground_truth = label[j] == i
            # predicted_mask = np.argmax(img[j], axis=0) == i
            intersection = np.logical_and(ground_truth, predicted_mask).sum()
            union = np.logical_or(ground_truth, predicted_mask).sum()
            IoU = intersection / union
            if not np.isnan(IoU):
                output[i]["iou"].append(IoU)
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
loader = DataLoader(dataset, batch_size=25, shuffle=True)
# loader = DataLoader(dataset, batch_size=5, shuffle=False)
IoU = {i: {"iou": []} for i in range(7)}
with torch.no_grad():
    for data in loader:
        images, labels, img = data[0].to(device), data[1].to(device), data[2]
        images = images.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.LongTensor).cuda()

        prediction = seg(images)
        prediction_mask = torch.argmax(prediction, 1)
        temp = getIOU(prediction_mask, labels)
        for i in range(7):
            IoU[i]["iou"] += temp[i]["iou"]

        # prediction_mask = torch.argmax(prediction, 1)
        # plt.title(MODEL_FILENAME)
        # plt.subplot(311)
        # plt.title("Images")
        # plt.imshow(np.transpose(
        #     make_grid(img, padding=5).cpu().detach().numpy(), (1, 2, 0)))
        # plt.subplot(312)
        # plt.title("Labels")
        # plt.imshow(convertLabel(labels))
        # plt.subplot(313)
        # plt.title("Predictions")
        # plt.imshow(convertPrediction(prediction_mask))
        # plt.show()
        # break

average = []
for i in range(7):
    average.append(np.average(IoU[i]["iou"]))
print("IoU", average)
print("average IoU", np.average(average))
