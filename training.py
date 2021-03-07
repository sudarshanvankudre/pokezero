import asyncio

from poke_env.server_configuration import LocalhostServerConfiguration

from players import MyRandomPlayer, PokeZero

num_games = 1

player1 = MyRandomPlayer(
    server_configuration=LocalhostServerConfiguration
)
player2 = MyRandomPlayer(
    server_configuration=LocalhostServerConfiguration
)

pokezero1 = PokeZero(
    server_configuration=LocalhostServerConfiguration
)

pokezero2 = PokeZero(
    server_configuration=LocalhostServerConfiguration
)


async def main():
    await pokezero1.battle_against(pokezero2, num_games)


asyncio.get_event_loop().run_until_complete(main())


def train(model, state_action_minibatch, value):
    """Trains the model on state_action_minibatch"""
    pass
