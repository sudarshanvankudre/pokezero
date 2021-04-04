import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader


def preprocessing(X, y):
    """Returns a DataLoader over X and y"""
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    train_data = []
    for i in range(len(X)):
        train_data.append([torch.unsqueeze(X[i], 0), y[i]])
    return DataLoader(train_data, batch_size=50, shuffle=True)


def learn(trainloader, model):
    num_epochs = 2
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()), lr=1e-5)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = torch.squeeze(model(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
