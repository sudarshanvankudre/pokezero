import asyncio

from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration

from players import MaxDamagePlayer

num_games = 100

player1 = RandomPlayer(
    server_configuration=LocalhostServerConfiguration
)
player2 = MaxDamagePlayer(
    server_configuration=LocalhostServerConfiguration
)


async def main():
    await player1.battle_against(player2, num_games)


asyncio.get_event_loop().run_until_complete(main())
