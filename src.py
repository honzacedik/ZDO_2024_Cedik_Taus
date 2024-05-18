import os
import cv2
import numpy as np
from skimage import segmentation, color, filters, io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def loadImages(path):
    images = {}

    for filename in os.listdir(path):

        if(filename == ".ipynb_checkpoints"):
          continue
        img = cv2.imread(os.path.join(path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGRhape)
        images[filename] = img

    return images

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Konvoluční vrstvy
        self.conv1 = nn.Conv2d(3, 21, kernel_size=(3), padding=1)
        self.conv2 = nn.Conv2d(21, 21, kernel_size=(3),padding=1)
        self.conv3 = nn.Conv2d(21, 21, kernel_size=(1),padding=1)
        self.conv4 = nn.Conv2d(21, 7, kernel_size=(1),padding=1)
        self.bn1 = nn.BatchNorm2d(21)
        self.bn2 = nn.BatchNorm2d(21)
        self.dropout = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool2d((2,5), stride=4, padding=1)
        self.pool2 = nn.MaxPool2d((3,3), stride=2, padding=1)


        # Plně propojené vrstvy
        self.fc1 = nn.LazyLinear(49)
        self.fc2 = nn.Linear(49,49)
        self.fc3 = nn.Linear(49,49)
        self.fc4 = nn.Linear(49, 7)


    def forward(self, x):
        # Přední průchod
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim = 1)
        return x