# -*- coding: utf-8 -*-
import argparse
import asyncio

import torch
from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration

from players import PokeZeroField, MaxDamagePlayer

parser = argparse.ArgumentParser(
    description="Use convolutional or fully connected model")
parser.add_argument('-conv', help="use convolutional model",
                    action="store_true")
args = parser.parse_args()
if args.conv:
    print("using saved convolutional model")
    net = torch.load("conv_model.pt")
else:
    print("using saved fully connected model")
    net = torch.load("fc_model.pt")

pokezero = PokeZeroField(
    server_configuration=LocalhostServerConfiguration,
    net=net
)

max_damage_player = MaxDamagePlayer(
    battle_format="gen8randombattle"
)

random_player = RandomPlayer(
    battle_format="gen8randombattle"
)

num_games = 100


async def main():
    await pokezero.battle_against(random_player, num_games)


asyncio.get_event_loop().run_until_complete(main())
print(
    "Pokezero won {}/{} battles".format(pokezero.n_won_battles, num_games)
)
