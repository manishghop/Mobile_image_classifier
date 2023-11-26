import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size=64

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32 , kernel_size = 3 ,padding = 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64 , kernel_size = 3 ,padding = 1)
        self.conv3 = nn.Conv2d(64, 64 , kernel_size = 3 ,padding = 1)
        self.conv4 = nn.Conv2d(64, 64 , kernel_size = 3 ,padding = 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(65536 , 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x=  self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.flatten(x) # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
