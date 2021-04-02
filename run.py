from poke_env.server_configuration import LocalhostServerConfiguration

from data_collection import Arena
from model import ConvPokeNet

net = ConvPokeNet()
arena = Arena(net, LocalhostServerConfiguration)

arena.play_n_games(5)
arena.save_dataset()
arena.evaluate()
