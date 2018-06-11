import torch.nn as nn
import torchvision.models as models
from Data import Data
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import pickle
import sys
import numpy as np
from SegNet import SegNet

TRAIN_DATA = "../mscoco_subset_cs601r/train/"
TRAIN_DATA_LABEL = "../mscoco_subset_cs601r/train_masks/"
TEST_DATA = "../mscoco_subset_cs601r/test/"
TEST_DATA_LABEL = "../mscoco_subset_cs601r/test_masks/"

# MODEL_FILENAME = 'model.json'
# LOSS_FILENAME = 'running_loss.pkl'
# MODEL_FILENAME = 'model2.json'
# LOSS_FILENAME = 'losses.pkl'
# MODEL_FILENAME = 'model3.json'
# LOSS_FILENAME = 'losses3.pkl'
MODEL_FILENAME = 'model4.json'
LOSS_FILENAME = 'losses4.pkl'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
seg = SegNet()
seg.to(device)
try:
    seg.load_state_dict(torch.load(MODEL_FILENAME))
except:
    pass

# datasets
train_dataset = Data(TRAIN_DATA, TRAIN_DATA_LABEL)
train_loader = DataLoader(train_dataset, batch_size=20,
                          shuffle=True, num_workers=10)

test_dataset = Data(TEST_DATA, TEST_DATA_LABEL)
test_loader = DataLoader(test_dataset, batch_size=10,
                         shuffle=True, num_workers=10)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(seg.parameters(), lr=5e-4, momentum=0.9)
optimizer = optim.Adam(seg.parameters(), lr=5e-4)
try:
    with open(LOSS_FILENAME, 'rb') as f:
        losses = pickle.load(f)
        running_loss = losses[0]
        test_loss = losses[1]
except:
    running_loss = []
    test_loss = []


def train(epoch_training_loss):
    for i, data in enumerate(train_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        images = images.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = seg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        loss_val = loss.item()
        # print(i, ":", loss_val)
        epoch_training_loss.append(loss_val)


def test(epoch_testing_loss):
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            images = images.type(torch.FloatTensor).cuda()
            labels = labels.type(torch.LongTensor).cuda()

            prediction = seg(images)
            loss = criterion(prediction, labels)
            epoch_testing_loss.append(loss.item())


try:
    epochs = int(sys.argv[1])
except:
    raise ValueError("No epoch value given")

very_start = time.time()
average_test_losses = [2.0]
for epoch in range(1, epochs+1):
    start = time.time()
    epoch_training_loss = []
    epoch_testing_loss = []

    train(epoch_training_loss)
    test(epoch_testing_loss)

    print('[Epoch:', epoch, '] [Time:',
          (time.time()-start)/60, " minutes] [Average train loss:",
          np.average(epoch_training_loss), "] [Average test loss:",
          np.average(epoch_testing_loss), "]")
    running_loss += epoch_training_loss
    test_loss += epoch_testing_loss
    torch.save(seg.state_dict(), MODEL_FILENAME)
    with open(LOSS_FILENAME, 'wb') as f:
        pickle.dump((running_loss, test_loss), f)
    # if average_test_losses[-1] < np.average(epoch_testing_loss):
    #     break
    average_test_losses.append(np.average(epoch_testing_loss))
print('Finished Training in ', (time.time()-very_start)/60, " minutes")
print("length of loss array: ", len(running_loss))
