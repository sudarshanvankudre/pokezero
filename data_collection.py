import torch
import asyncio
from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from poke_env.server_configuration import ServerConfiguration
from tabulate import tabulate
from torch import nn

from players import MaxDamagePlayer, PokeZeroStudent


class Arena():
    random_player = RandomPlayer()
    max_damage_player = MaxDamagePlayer()

    def __init__(self, m: nn.Module, server_config: ServerConfiguration):
        self.player1 = PokeZeroStudent(server_config, m)
        self.player2 = PokeZeroStudent(server_config, m)
        self.model = m

    async def play(self):
        await self.player1.battle_against(self.player2, 1)

    def play_n_games(self, n):
        for _ in range(n):
            asyncio.get_event_loop().run_until_complete(self.play())


    def save_model(self):
        torch.save(self.model, "promising_models/current_model.pt")

    async def evaluate(self):
        players = [self.player1, self.random_player, self.max_damage_player]
        cross_evaluation = await cross_evaluate(players, n_challenges=20)
        # Defines a header for displaying results
        table = [["-"] + [p.username for p in players]]

        # Adds one line per player with corresponding results
        for p_1, results in cross_evaluation.items():
            table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

        # Displays results in a nicely formatted table.
        print(tabulate(table))
