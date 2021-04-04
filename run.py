from poke_env.server_configuration import LocalhostServerConfiguration

from clustering import get_model_training_data, transform_dataset
from data_collection import Arena
from learning import learn, preprocessing
from model import ConvPokeNet

training_cycles = 1
net = ConvPokeNet()
arena = Arena(net, LocalhostServerConfiguration)

for _ in range(training_cycles):
    arena.play_n_games(100)
    dataset = arena.save_dataset()
    X, y = transform_dataset(dataset)
    model_input, model_labels = get_model_training_data(X, y)
    trainloader = preprocessing(model_input, model_labels)
    learn(trainloader, net)
