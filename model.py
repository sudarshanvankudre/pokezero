import torch
from torch import nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    def __init__(self):
        super(ConvPokeNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 9)
        self.conv2 = nn.Conv1d(64, 64, 9)
        self.pool1 = nn.MaxPool1d(10)
        self.conv3 = nn.Conv1d(64, 128, 9)
        self.conv4 = nn.Conv1d(128, 128, 9)
        self.pool2 = nn.MaxPool1d(6)
        self.conv5 = nn.Conv1d(128, 256, 9)
        self.conv6 = nn.Conv1d(256, 256, 9)
        self.pool3 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(9472, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        n = x.shape[0]
        x = torch.reshape(x, (n, x.shape[1] * x.shape[2]))
        x = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))
        return x
