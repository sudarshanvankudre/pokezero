import torch
from torch import nn
from torch.nn import functional as F


class FCPokeNet(nn.Module):
    def __init__(self):
        super(FCPokeNet, self).__init__()
        self.fc1 = nn.Linear(10197, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ConvPokeNet(nn.Module):
    def __init__(self, filter1=16, filter2=32, linear1=4000, kernel_size=9):
        super(ConvPokeNet, self).__init__()
        self.conv1 = nn.Conv1d(1, filter1, kernel_size)
        self.conv2 = nn.Conv1d(filter1, filter2, kernel_size)
        self.pool = nn.MaxPool1d(10)
        self.conv3 = nn.Conv1d(filter2, filter2, kernel_size)
        self.conv4 = nn.Conv1d(filter2, filter2, kernel_size)
        self.fc1 = nn.Linear(3840, linear1)
        self.fc2 = nn.Linear(linear1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        n = x.shape[0]
        x = torch.reshape(x, (n, x.shape[1] * x.shape[2]))
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
