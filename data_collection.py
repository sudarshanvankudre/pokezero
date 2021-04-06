import asyncio
import pickle

from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from poke_env.server_configuration import ServerConfiguration
from tabulate import tabulate
from torch import nn

from players import MaxDamagePlayer, PokeZeroStudent


class Arena():
    random_player = RandomPlayer()
    max_damage_player = MaxDamagePlayer()

    def __init__(self, model: nn.Module, server_config: ServerConfiguration):
        self.player1 = PokeZeroStudent(server_config, model)
        self.player2 = PokeZeroStudent(server_config, model)
        self.model = model
        self.dataset = dict()
        self.p1_battles_won = 0
        self.p2_battles_won = 0
        self.dataset_num = 0

    async def play(self):
        await self.player1.battle_against(self.player2, 1)

    def update_dataset(self):
        if self.player1.n_won_battles == self.p1_battles_won + 1:
            for gs_action in self.player1.gs_actions:
                self.dataset[gs_action] = 1
            for gs_action in self.player2.gs_actions:
                self.dataset[gs_action] = 0
            self.p1_battles_won += 1
        elif self.player2.n_won_battles == self.p2_battles_won + 1:
            for gs_action in self.player1.gs_actions:
                self.dataset[gs_action] = 0
            for gs_action in self.player2.gs_actions:
                self.dataset[gs_action] = 1
            self.p2_battles_won += 1
        else:
            for gs_action in self.player1.gs_actions:
                self.dataset[gs_action] = 0
            for gs_action in self.player2.gs_actions:
                self.dataset[gs_action] = 0
        self.player1.gs_actions = []
        self.player2.gs_actions = []

    def play_n_games(self, n):
        for i in range(n):
            if i % 10 == 0:
                print("Game {}".format(i))
            asyncio.get_event_loop().run_until_complete(self.play())
            self.update_dataset()

    def save_dataset(self):
        with open("datasets/dataset{}.pickle".format(self.dataset_num), 'wb') as fout:
            pickle.dump(self.dataset, fout)
        self.dataset_num += 1
        return self.dataset

    async def _evaluate_helper(self):
        num_games = 10

    def evaluate(self):
        print("Evaluating performance")
        asyncio.get_event_loop().run_until_complete(self._evaluate_helper())
