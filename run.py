import time

from poke_env.server_configuration import LocalhostServerConfiguration

from clustering import *
from data_collection import Arena
from model import ConvPokeNet

net = ConvPokeNet()
arena = Arena(net, LocalhostServerConfiguration)

arena.play_n_games(100)
arena.save_dataset()
# arena.evaluate()

X, y = load_dataset(0)
start_time = time.time()
labels = labels_array(X)
win_rate = win_rates(labels, y)
print("--- %s seconds ---" % (time.time() - start_time))
