# This file is used to deploy an agent onto play.pokemonshowdown.com to play against humans

import asyncio

from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from players import MaxDamagePlayer, MyRandomPlayer

username = "pokkezero"
password = "5&HKKFVQ9Pznd!!unYtHOyke@*SZvzMyPrdCqaEWy9wRb3tBe*Ch0r*KI$GTAoWvxhAEe#9p8aJ&VUgthEf2WgCH6Dwg^odMBOa"
num_games = 2

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


async def main():
    await my_random_player.ladder(num_games)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
