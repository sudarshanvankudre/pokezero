import asyncio

from poke_env.server_configuration import LocalhostServerConfiguration

from players import MyRandomPlayer

num_games = 1

player1 = MyRandomPlayer(
    server_configuration=LocalhostServerConfiguration
)
player2 = MyRandomPlayer(
    server_configuration=LocalhostServerConfiguration
)


async def main():
    await player1.battle_against(player2, num_games)


asyncio.get_event_loop().run_until_complete(main())
