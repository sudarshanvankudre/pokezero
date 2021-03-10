import argparse
import asyncio

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from poke_env.server_configuration import LocalhostServerConfiguration
from torch import nn

from model import FCPokeNet, ConvPokeNet
from players import PokeZero

parser = argparse.ArgumentParser(description='Use new or continue from saved')
parser.add_argument('-n', '--new', help="use brand new model", action="store_true")
parser.add_argument('-fc', help="use fully connected model", action="store_true")
parser.add_argument('-conv', help="use convolutional network", action="store_true")
args = parser.parse_args()

if args.new:
    print("initializing new model")
    if args.fc:
        print("using fully_connected_model")
        net = FCPokeNet()
    elif args.conv:
        print("using convolutional model")
        net = ConvPokeNet()
else:
    print("using saved model")
    if args.fc:
        print("using fully_connected_model")
        net = torch.load("fc_model.pt")
    elif args.conv:
        print("using convolutional model")
        net = torch.load("conv_model.pt")

num_games = 1

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


def play_train_loop(n=1, model=None, training_cycles=1, model_type="conv"):
    p1_battles_won = 0
    p2_battles_won = 0
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    losses = []
    running_avg_loss = 0
    optimizer.zero_grad()
    for cycle in range(training_cycles):
        print("battle {}".format(cycle))
        # play
        try:
            for _ in range(n):
                asyncio.get_event_loop().run_until_complete(main())

            # update predictions
            # print("updating predictions")
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
            # print("preprocessing new data")
            inputs = torch.empty(len(pokezero1.predictions) + len(pokezero2.predictions), 1, gs_action.shape[2])
            labels = torch.empty(len(pokezero1.predictions) + len(pokezero2.predictions))
            i = 0
            for k, v in pokezero1.predictions.items():
                inputs[i] = k
                labels[i] = float(v)
                i += 1
            for k, v in pokezero2.predictions.items():
                inputs[i] = k
                labels[i] = float(v)
                i += 1
            for epoch in range(1):
                # print('calculating outputs...')
                # x = torch.unsqueeze(torch.from_numpy(np.array(inputs)).float(), 1)
                outputs = model(inputs)
                loss = criterion(outputs, torch.Tensor(labels).reshape_as(outputs))
                losses.append(float(loss))
                running_avg_loss = (running_avg_loss + float(loss)) / len(losses)
                print("Running average loss:", running_avg_loss)
                # print(loss)
                # print("loss", loss)
                # print('backpropagating...')
                loss.backward()
                # print('optimizing')
                optimizer.step()
        except Exception as e:
            print("cycle failed")
            raise e
        if model_type == "conv":
            torch.save(model, "conv_model.pt")
        elif model_type == "fc":
            torch.save(model, "fc_model.pt")
        print('model saved')
    plt.plot(losses)
    plt.show()


play_train_loop(model=net, training_cycles=10)
