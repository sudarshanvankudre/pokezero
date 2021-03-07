import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PokeNet(nn.Module):
    def __init__(self):
        super(PokeNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 9)
        self.pool1 = nn.MaxPool1d(9, stride=4)
        self.conv2 = nn.Conv1d(64, 128, 9)
        self.conv3 = nn.Conv1d(128, 128, 9)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.fc2(F.relu(self.fc1(x)))

