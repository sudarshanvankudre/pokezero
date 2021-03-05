import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PokeNet(nn.Module):
    def __init__(self):
        super(PokeNet, self).__init__()
        self.avg_pool1 = nn.AvgPool1d(7, stride=3)
        self.conv1 = nn.Conv1d()