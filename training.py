import argparse
import asyncio

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from poke_env.server_configuration import LocalhostServerConfiguration
from torch import nn
from torch.optim.lr_scheduler import StepLR

from model import FCPokeNet, ConvPokeNet, ResPokeNet
from players import PokeZeroTrain

# default is fc options are others
parser = argparse.ArgumentParser(description='Use new or continue from saved')
parser.add_argument('--new', help="use brand new model", action="store_true")
parser.add_argument('-res', help="use residual model", action="store_true")
parser.add_argument(
    '-conv', help="use convolutional network", action="store_true")
parser.add_argument('-c', help="number of training battles",
                    default=1000, type=int)
args = parser.parse_args()

if args.new:
    if args.res:
        print("using new residual model")
        net = ResPokeNet(1)
        model_type = "res"
    elif args.conv:
        print("using new convolutional model")
        net = ConvPokeNet()
        model_type = "conv"
    else:
        print("using new fully connected model")
        net = FCPokeNet()
        model_type = "fc"
else:
    if args.res:
        print("using saved residual model")
        net = torch.load("res_model.pt")
        model_type = "res"
    elif args.conv:
        print("using saved convolutional model")
        net = torch.load("conv_model.pt")
        model_type = "conv"
    else:
        print("using saved fully connected model")
        net = torch.load("fc_model.pt")
        model_type = "fc"

num_games = 1

pokezero1 = PokeZeroTrain(
    server_configuration=LocalhostServerConfiguration,
    net=net
)

pokezero2 = PokeZeroTrain(
    server_configuration=LocalhostServerConfiguration,
    net=net
)


async def main():
    await pokezero1.battle_against(pokezero2, num_games)


def update_predictions(p1, p2, p1_battles_won, p2_battles_won):
    if p1.n_won_battles == p1_battles_won + 1:
        for gs_action in p1.predictions:
            p1.predictions[gs_action] += 1
        for gs_action in p2.predictions:
            p2.predictions[gs_action] -= 1
        p1_battles_won = p1.n_won_battles
    elif p2.n_won_battles == p2_battles_won + 1:
        for gs_action in p2.predictions:
            p2.predictions[gs_action] += 1
        for gs_action in p1.predictions:
            p1.predictions[gs_action] -= 1
        p2_battles_won = p2.n_won_battles
    return p1_battles_won, p2_battles_won, gs_action.shape[2]


def prepare_inputs_and_labels(p1, p2, feature_shape):
    inputs = torch.empty((len(p1.predictions) +
                          len(p2.predictions), 2, feature_shape))
    labels = torch.empty(
        len(p1.predictions) + len(p2.predictions))
    i = 0
    for k, v in p1.predictions.items():
        inputs[i] = k
        labels[i] = float(v)
        i += 1
    for k, v in p2.predictions.items():
        inputs[i] = k
        labels[i] = float(v)
        i += 1
    p1.predictions.clear()
    p2.predictions.clear()
    return inputs, labels


def training_step(model, inputs, labels, loss_fn, optimizer, scheduler):
    model.train()
    outputs = torch.squeeze(model(inputs))
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def graph_losses(losses):
    plt.plot(losses)
    plt.xlabel("Battles played")
    plt.ylabel("Loss")
    plt.show()
    plt.hist(losses)
    plt.show()


def play_train_loop(model=None, training_cycles=1, model_type="fc"):
    p1_battles_won = 0
    p2_battles_won = 0
    loss_fn = nn.MSELoss(reduction="sum")
    # optimizer = optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
    # optimizer = optim.Adadelta(model.parameters())
    optimizer = optim.Adam(list(model.parameters()), lr=1e-4)
    scheduler = StepLR(optimizer, 100)
    losses = []
    for cycle in range(training_cycles):
        print("battle {}".format(cycle))
        # play
        try:
            asyncio.get_event_loop().run_until_complete(main())

            p1_battles_won, p2_battles_won, feature_shape = update_predictions(pokezero1, pokezero2, p1_battles_won,
                                                                               p2_battles_won)
            inputs, labels = prepare_inputs_and_labels(
                pokezero1, pokezero2, feature_shape)
            loss = training_step(model, inputs, labels,
                                 loss_fn, optimizer, scheduler)
            print(loss)
            losses.append(loss)
        except Exception as e:
            print("cycle failed")
            raise e
        if model_type == "conv":
            torch.save(model, "conv_model.pt")
            print("model saved to {}_model.pt".format(model_type))
        elif model_type == "fc":
            torch.save(model, "fc_model.pt")
            print("model saved to {}_model.pt".format(model_type))
        elif model_type == "res":
            torch.save(model, "res_model.pt")
            print(f"model saved to {model_type}_model.pt")

    graph_losses(losses)


play_train_loop(model=net, training_cycles=args.c, model_type=model_type)
