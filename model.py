import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PokeNet(nn.Module):
    def __init__(self):
        super(PokeNet, self).__init__()
        # self.conv1 = nn.Conv1d(1, 64, 21)
        # self.pool1 = nn.MaxPool1d(9, stride=4)
        # self.conv2 = nn.Conv1d(64, 128, 11)
        # self.conv3 = nn.Conv1d(128, 256, 9)
        # self.conv4 = nn.Conv1d(256, 256, 9)
        # self.fc1 = nn.Linear(128, 256)
        # self.fc2 = nn.Linear(256, 1)
        self.fc1 = nn.Linear(10197, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
