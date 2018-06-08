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
        self.deconvRelu1 = nn.ReLU()
        self.deconvRelu2 = nn.ReLU()
        self.deconvRelu3 = nn.ReLU()
        self.deconvRelu4 = nn.ReLU()
        self.deconvRelu5 = nn.ReLU()

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
        # print("Starting size: {}".format(x.size()))
        x = self.deconv1(x)
        x = self.deconvRelu1(x)
        # print("Deconv1 size: {}".format(x.size()))
        x = self.deconv2(x)
        x = self.deconvRelu2(x)
        # print("Deconv2 size: {}".format(x.size()))
        x = self.deconv3(x)
        x = self.deconvRelu3(x)
        # print("Deconv3 size: {}".format(x.size()))
        x = self.deconv4(x)
        x = self.deconvRelu4(x)
        # print("Deconv4 size: {}".format(x.size()))
        x = self.deconv5(x,  output_size=original_size)
        x = self.deconvRelu5(x)
        # print("Deconv5 size: {}".format(x.size()))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)
#     seg = SegNet()
#     seg.to(device)
#     try:
#         seg.load_state_dict(torch.load('model.json'))
#     except:
#         pass
#     dataset = Data(TRAIN_DATA, TRAIN_DATA_LABEL)
#     loader = DataLoader(dataset, batch_size=22, shuffle=True, num_workers=10)

#     criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.SGD(seg.parameters(), lr=5e-4, momentum=0.9)
#     optimizer = optim.Adam(seg.parameters(), lr=5e-4)
#     try:
#         with open('running_loss.pkl', 'rb') as f:
#             running_loss = pickle.load(f)
#     except:
#         running_loss = []

#     very_start = time.time()
#     for epoch in range(1, 4):
#         start = time.time()
#         epoch_loss = []
#         for i, data in enumerate(loader, 0):
#             images, labels = data[0].to(device), data[1].to(device)
#             images = images.type(torch.FloatTensor).cuda()
#             labels = labels.type(torch.LongTensor).cuda()
#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = seg(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             loss_val = loss.item()
#             # print(i, ":", loss_val)
#             epoch_loss.append(loss_val)

#         print('[Epoch:', epoch, '] [Time:',
#               (time.time()-start)/60, " minutes] [Average loss:", np.average(epoch_loss), "]")
#         running_loss += epoch_loss
#         torch.save(seg.state_dict(), 'model.json')
#         with open('running_loss.pkl', 'wb') as f:
#             pickle.dump(running_loss, f)
# print('Finished Training in ', (time.time()-very_start)/60, " minutes")
# print("length of loss array: ", len(running_loss))

# # notes:
# #  - ask about stride and padding
# #  - research or ask about the crossentropy
# #       - change labels to be single channel with all values converted to a 1-6
# #       - at each pixel location of the output you should get an array of 7 length that has the probabilities of all the categories
