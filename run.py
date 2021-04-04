import argparse

import torch
from poke_env.server_configuration import LocalhostServerConfiguration

from clustering import get_model_training_data, transform_dataset
from data_collection import Arena
from learning import learn, preprocessing
from model import ConvPokeNet

parser = argparse.ArgumentParser(description='Use new model or continue from saved')
parser.add_argument('--new', help="use brand new model", action="store_true")
args = parser.parse_args()

training_cycles = 20
if args.new:
    net = ConvPokeNet()
else:
    net = torch.load("poke_conv.pt")
arena = Arena(net, LocalhostServerConfiguration)

for cycle in range(training_cycles):
    print("Training cycle {}".format(cycle))
    arena.play_n_games(100)
    dataset = arena.save_dataset()
    print("Massaging data...")
    X, y = transform_dataset(dataset)
    model_input, model_labels = get_model_training_data(X, y)
    trainloader = preprocessing(model_input, model_labels)
    print("Learning...")
    learn(trainloader, net)
    torch.save(net, "poke_conv.pt")
    print("Evaluation")
    arena.evaluate()