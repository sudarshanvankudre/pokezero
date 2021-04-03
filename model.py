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
    def __init__(self, filter1=16, filter2=32, filter3=64, linear1=1000, linear2=500, linear3=100):
        super(ConvPokeNet, self).__init__()
        self.conv1 = nn.Conv1d(1, filter1, 9)
        self.conv2 = nn.Conv1d(filter1, filter1, 9)
        self.pool1 = nn.MaxPool1d(10)
        self.conv3 = nn.Conv1d(filter1, filter2, 9)
        self.conv4 = nn.Conv1d(filter2, filter2, 9)
        self.pool2 = nn.MaxPool1d(6)
        self.conv5 = nn.Conv1d(filter2, filter3, 9)
        self.conv6 = nn.Conv1d(filter3, filter3, 9)
        self.pool3 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(3136, linear1)
        self.batchnorm1 = nn.BatchNorm1d(linear1)
        self.fc2 = nn.Linear(linear1, linear2)
        self.batchnorm2 = nn.BatchNorm1d(linear2)
        self.fc3 = nn.Linear(linear2, linear3)
        self.batchnorm3 = nn.BatchNorm1d(linear3)
        self.fc4 = nn.Linear(linear3, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        n = x.shape[0]
        x = torch.reshape(x, (n, x.shape[1] * x.shape[2]))
        x = self.fc4(F.relu(self.batchnorm3(self.fc3(
            F.relu(self.batchnorm2(self.fc2(F.relu(self.batchnorm1(self.fc1(x))))))))))
        return x


class ResLayer(nn.Module):
    def __init__(self, filter_size, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(filter_size, filter_size, kernel_size)
        self.pool1 = nn.MaxPool1d(2)
        self.batchnorm = nn.BatchNorm1d(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size)
        self.pool2 = nn.MaxPool1d(4)

    def forward(self, x):
        res = x
        x = F.relu(self.batchnorm(self.pool1(self.conv1(x))))
        x = self.batchnorm(self.pool2(self.conv2(x)))
        x += F.interpolate(res, x.shape[2])
        return F.relu(x)


class ConvLayer(nn.Module):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(filter_in, filter_out, kernel_size, bias=False)
        self.batchnorm = nn.BatchNorm1d(filter_out)

    def forward(self, x):
        return F.relu(self.batchnorm(self.conv(x)))


class ResPokeNet(nn.Module):
    def __init__(self, num_res_layers):
        super(ResPokeNet, self).__init__()
        self.conv_layer = ConvLayer(2, 64, 9)
        self.res_layer = ResLayer(64, 9)
        self.num_res_layers = num_res_layers
        self.value_head = ValueHead(64)

    def forward(self, x):
        x = self.conv_layer(x)
        for i in range(self.num_res_layers):
            x = self.res_layer(x)
        return self.value_head(x)


class ValueHead(nn.Module):
    def __init__(self, filter_in):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv1d(filter_in, 1, 9, bias=False)
        self.batchnorm = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(1600, 1000)
        self.fc2 = nn.Linear(1000, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.batchnorm(self.conv1(x)))
        n = x.shape[0]
        x = torch.reshape(x, (n, x.shape[1] * x.shape[2]))
        x = self.tanh(self.fc2(F.relu(self.fc1(x))))
        return x
