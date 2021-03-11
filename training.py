import argparse
import asyncio

import torch
import torch.optim as optim
from poke_env.server_configuration import LocalhostServerConfiguration
from torch import nn

from model import FCPokeNet, ConvPokeNet
from players import PokeZero

parser = argparse.ArgumentParser(description='Use new or continue from saved')
parser.add_argument(
    '-n', '--new', help="use brand new model", action="store_true")
parser.add_argument('-fc', help="use fully connected model",
                    action="store_true")
parser.add_argument(
    '-conv', help="use convolutional network", action="store_true")
args = parser.parse_args()

if args.new:
    if args.fc:
        print("using new fully_connected_model")
        net = FCPokeNet()
        model_type = "fc"
    elif args.conv:
        print("using new convolutional model")
        net = ConvPokeNet()
        model_type = "conv"
else:
    if args.fc:
        print("using saved fully_connected_model")
        net = torch.load("fc_model.pt")
        model_type = "fc"
    elif args.conv:
        print("using saved convolutional model")
        net = torch.load("conv_model.pt")
        model_type = "conv"

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


def play_train_loop(model=None, training_cycles=1, model_type="conv", batch_size=16):
    p1_battles_won = 0
    p2_battles_won = 0
    loss_fn = nn.MSELoss(reduction="sum")
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(list(model.parameters()), lr=1e-5)
    for cycle in range(training_cycles):
        print("battle {}".format(cycle))
        # play
        try:
            asyncio.get_event_loop().run_until_complete(main())

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
            inputs = torch.empty((len(pokezero1.predictions) +
                                  len(pokezero2.predictions), 1, gs_action.shape[2]))
            labels = torch.empty(
                len(pokezero1.predictions) + len(pokezero2.predictions))
            i = 0
            for k, v in pokezero1.predictions.items():
                inputs[i] = k
                labels[i] = float(v)
                i += 1
            for k, v in pokezero2.predictions.items():
                inputs[i] = k
                labels[i] = float(v)
                i += 1
            pokezero1.predictions.clear()
            pokezero2.predictions.clear()

            outputs = torch.squeeze(model(inputs))
            loss = loss_fn(outputs, labels)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except Exception as e:
            print("cycle failed")
            raise e
        if model_type == "conv":
            torch.save(model, "conv_model.pt")
            print("model saved to {}_model.pt".format(model_type))
        elif model_type == "fc":
            torch.save(model, "fc_model.pt")
            print("model saved to {}_model.pt".format(model_type))


play_train_loop(model=net, training_cycles=1000, model_type=model_type)
