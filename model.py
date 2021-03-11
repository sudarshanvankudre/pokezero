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
    def __init__(self, filter1=16, filter2=32, filter3=64, linear1=1000, linear2=500, linear3=100, stride=10):
        super(ConvPokeNet, self).__init__()
        self.batch_norm = nn.BatchNorm1d(1)
        self.conv1 = nn.Conv1d(1, filter1, 9)
        self.conv2 = nn.Conv1d(filter1, filter1, 9)
        self.pool1 = nn.MaxPool1d(10)
        self.conv3 = nn.Conv1d(filter1, filter2, 9)
        self.conv4 = nn.Conv1d(filter2, filter2, 9)
        self.pool2 = nn.MaxPool1d(6)
        self.conv5 = nn.Conv1d(filter2, filter3, 9)
        self.conv6 = nn.Conv1d(filter3, filter3, 9)
        self.pool3 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2880, linear1)
        self.fc2 = nn.Linear(linear1, linear2)
        self.fc3 = nn.Linear(linear2, linear3)
        self.fc4 = nn.Linear(linear3, 1)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        n = x.shape[0]
        x = torch.reshape(x, (n, x.shape[1] * x.shape[2]))
        x = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))
        return x
