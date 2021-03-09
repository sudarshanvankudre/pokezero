import asyncio

import numpy as np
import torch
import torch.optim as optim
from poke_env.server_configuration import LocalhostServerConfiguration
from torch import nn

from model import PokeNet
from players import MyRandomPlayer, PokeZero

num_games = 1
net = PokeNet()

player1 = MyRandomPlayer(
    server_configuration=LocalhostServerConfiguration
)
player2 = MyRandomPlayer(
    server_configuration=LocalhostServerConfiguration
)

pokezero1 = PokeZero(
    server_configuration=LocalhostServerConfiguration,
    net=net
)

pokezero2 = PokeZero(
    server_configuration=LocalhostServerConfiguration,
    net=net
)


async def main():
    await pokezero1.battle_against(pokezero2, num_games)


def play_train_loop(p1=None, p2=None, n=1, model=None, training_cycles=1):
    p1_battles_won = 0
    p2_battles_won = 0
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for _ in range(training_cycles):
        # play
        print('beginning play')
        for _ in range(n):
            asyncio.get_event_loop().run_until_complete(main())

        # update predictions
        if pokezero1.n_won_battles == p1_battles_won + 1:
            for gs_action in pokezero1.predictions:
                pokezero1.predictions[gs_action] += 1
            for gs_action in pokezero2.predictions:
                pokezero2.predictions[gs_action] -= 1
            p1_battles_won = pokezero1.n_won_battles
        elif pokezero2.n_won_battles == p2_battles_won + 1:
            for gs_action in pokezero2.predictions:
                pokezero2.predictions[gs_action] += 1
            for gs_action in pokezero1.predictions:
                pokezero1.predictions[gs_action] -= 1
            p2_battles_won = pokezero2.n_won_battles

        # get training data
        inputs = np.empty((len(pokezero1.predictions) + len(pokezero2.predictions), gs_action.shape[0]))
        labels = np.empty(len(pokezero1.predictions) + len(pokezero2.predictions))
        i = 0
        for k, v in pokezero1.predictions.items():
            inputs[i] = k
            labels[i] = v
            i += 1
        for k, v in pokezero2.predictions.items():
            inputs[i] = k
            labels[i] = v
            i += 1
        for epoch in range(1):
            print(epoch)
            optimizer.zero_grad()
            print('calculating outputs...')
            outputs = model(torch.from_numpy(np.array(inputs)).float())
            loss = criterion(outputs, torch.Tensor(labels).reshape_as(outputs))
            print('backpropagating...')
            loss.backward()
            optimizer.step()


play_train_loop(model=net, training_cycles=2)


def learn(model, x, y):
    """Trains model on x, y data and returns modified model"""
    pass
