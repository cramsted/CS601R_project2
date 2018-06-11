import torch.nn as nn
import torchvision.models as models
from Data import Data
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import pickle
import numpy as np


TRAIN_DATA = "../mscoco_subset_cs601r/train/"
TRAIN_DATA_LABEL = "../mscoco_subset_cs601r/train_masks/"
TEST_DATA = "../mscoco_subset_cs601r/test/"
TEST_DATA_LABEL = "../mscoco_subset_cs601r/test_masks/"


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(
            2048, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            512, 128, 4, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, 5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(
            64, 32, 4, stride=2, padding=3, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(
            32, 7, 4, stride=2, padding=2, output_padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight)
        nn.init.xavier_uniform_(self.deconv2.weight)
        nn.init.xavier_uniform_(self.deconv3.weight)
        nn.init.xavier_uniform_(self.deconv4.weight)
        nn.init.xavier_uniform_(self.deconv5.weight)
        self.deconvRelu1 = nn.LeakyReLU()
        self.deconvRelu2 = nn.LeakyReLU()
        self.deconvRelu3 = nn.LeakyReLU()
        self.deconvRelu4 = nn.LeakyReLU()
        self.deconvRelu5 = nn.LeakyReLU()

    def forward(self, x):
        original_size = x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # deconvolution layers
        x = self.deconv1(x)
        x = self.deconvRelu1(x)
        x = self.deconv2(x)
        x = self.deconvRelu2(x)
        x = self.deconv3(x)
        x = self.deconvRelu3(x)
        x = self.deconv4(x)
        x = self.deconvRelu4(x)
        x = self.deconv5(x,  output_size=original_size)
        x = self.deconvRelu5(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
