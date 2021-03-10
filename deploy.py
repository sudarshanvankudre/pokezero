# This file is used to deploy an agent onto play.pokemonshowdown.com to play against humans

import asyncio

import torch
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from players import MaxDamagePlayer, MyRandomPlayer, PokeZeroField

username = "pokkezero"
password = "5&HKKFVQ9Pznd!!unYtHOyke@*SZvzMyPrdCqaEWy9wRb3tBe*Ch0r*KI$GTAoWvxhAEe#9p8aJ&VUgthEf2WgCH6Dwg^odMBOa"
num_games = 100

random_player = RandomPlayer(
    player_configuration=PlayerConfiguration(username, password),
    server_configuration=ShowdownServerConfiguration
)
max_dmg_player = MaxDamagePlayer(
    player_configuration=PlayerConfiguration(username, password),
    server_configuration=ShowdownServerConfiguration
)

my_random_player = MyRandomPlayer(
    player_configuration=PlayerConfiguration(username, password),
    server_configuration=ShowdownServerConfiguration
)

pokezero_field = PokeZeroField(
    player_configuration=PlayerConfiguration(username, password),
    server_configuration=ShowdownServerConfiguration,
    net=torch.load("fc_model.pt")
)


async def main():
    await pokezero_field.ladder(num_games)


asyncio.get_event_loop().run_until_complete(main())

print(f"Pokezero won {pokezero_field.n_won_battles}/{num_games} battles")
