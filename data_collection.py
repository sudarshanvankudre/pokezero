import asyncio
import pickle
import random

from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import ServerConfiguration
from torch import nn

from players import MaxDamagePlayer, PokeZeroStudent, PokeZeroEval


class Arena():
    random_player = RandomPlayer()
    max_damage_player = MaxDamagePlayer()

    def __init__(self, model: nn.Module, server_config: ServerConfiguration):
        self.epsilon = 1.0
        self.pokezero1 = PokeZeroStudent(server_config, model, self.epsilon)
        self.pokezero2 = PokeZeroStudent(server_config, model, self.epsilon)
        self.pokezero_eval = PokeZeroEval(server_config, model)
        self.model = model
        self.dataset = dict()
        self.p1_battles_won = 0
        self.p2_battles_won = 0
        self.dataset_num = 0
        self.decay = 1

    async def play(self):
        opponent = random.choice([self.pokezero2, self.random_player, self.max_damage_player])
        await self.pokezero1.battle_against(opponent, 1)

    def update_dataset(self):
        if self.pokezero1.n_won_battles == self.p1_battles_won + 1:
            for gs_action in self.pokezero1.gs_actions:
                self.dataset[gs_action] = 1
            for gs_action in self.pokezero2.gs_actions:
                self.dataset[gs_action] = 0
            self.p1_battles_won += 1
        elif self.pokezero2.n_won_battles == self.p2_battles_won + 1:
            for gs_action in self.pokezero1.gs_actions:
                self.dataset[gs_action] = 0
            for gs_action in self.pokezero2.gs_actions:
                self.dataset[gs_action] = 1
            self.p2_battles_won += 1
        else:
            for gs_action in self.pokezero1.gs_actions:
                self.dataset[gs_action] = 0
            for gs_action in self.pokezero2.gs_actions:
                self.dataset[gs_action] = 0
        self.pokezero1.gs_actions = []
        self.pokezero2.gs_actions = []

    def play_n_games(self, n):
        for i in range(n):
            if i % 10 == 0:
                print(f"Game {i}")
            self.pokezero1.epsilon = self.epsilon
            self.pokezero2.epsilon = self.epsilon
            asyncio.get_event_loop().run_until_complete(self.play())
            self.update_dataset()
            self.epsilon *= self.decay

    def save_dataset(self):
        with open("datasets/dataset{}.pickle".format(self.dataset_num), 'wb') as fout:
            pickle.dump(self.dataset, fout)
        self.dataset_num += 1
        return self.dataset

    async def _evaluate_helper(self):
        num_games = 10
        self.pokezero_eval.reset_battles()
        await self.pokezero_eval.battle_against(self.random_player, num_games)
        random_win_rate = self.pokezero_eval.n_won_battles / num_games
        with open("vs_random_results.txt", "w") as fout:
            fout.write(str(random_win_rate) + "\n")
        self.pokezero_eval.reset_battles()
        await self.pokezero_eval.battle_against(self.max_damage_player, num_games)
        max_damage_win_rate = self.pokezero_eval.n_won_battles / num_games
        with open("vs_max_damage_results.txt", "w") as fout:
            fout.write(str(max_damage_win_rate) + "\n")
        print(f"num games: {num_games}")
        print(f"vs random: {random_win_rate}")
        print(f"vs max damage: {max_damage_win_rate}")

    def evaluate(self):
        asyncio.get_event_loop().run_until_complete(self._evaluate_helper())
