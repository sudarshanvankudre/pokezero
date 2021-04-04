import random
from collections import Counter

import numpy as np
import torch
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from dataloader.preprocessing import game_state, move_name_onehot_vector, pokemon_species_onehot_vector
from stats import random_battle_total_pokemon, random_battle_total_moves

num_pokemon = random_battle_total_pokemon()
num_moves = random_battle_total_moves()


class MaxDamagePlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.available_moves:
            best_move = max(battle.available_moves,
                            key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


class MyRandomPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        print(battle.active_pokemon)
        return self.choose_random_move(battle)


def get_action_vector(action):
    if type(action) is Move:
        return move_name_onehot_vector(action.id)
    else:
        return pokemon_species_onehot_vector(action.species)


class PokeZero(Player):
    gs_action_vector_max_size = 12909

    def __init__(self, server_configuration, net, player_configuration=None):
        super(PokeZero, self).__init__(server_configuration=server_configuration,
                                       player_configuration=player_configuration)
        self.model = net
        # self.prev_gs_action = torch.empty(self.gs_action_vector_max_size)

    def get_model_input(self, gs, action_vector):
        model_input = np.concatenate((gs, action_vector))
        try:
            model_input = np.pad(model_input, (0, self.gs_action_vector_max_size - model_input.shape[0]), 'constant',
                                 constant_values=0)
        except ValueError as e:
            print(model_input.shape)
            raise e
        model_input = torch.from_numpy(model_input)
        # c = torch.stack((model_input, self.prev_gs_action))
        # self.prev_gs_action = model_input
        # model_input = torch.unsqueeze(c, 0)
        model_input = torch.unsqueeze(torch.unsqueeze(model_input, 0), 0)
        return model_input.float()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        pass


class PokeZeroStudent(PokeZero):
    def __init__(self, server_config, net):
        super().__init__(server_config, net)
        self.gs_actions = list()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self.model.eval()
        if battle.turn > 500:
            return self.choose_random_move(battle)
        gs = game_state(battle)
        best_action = None
        best_gs_action = None
        best_value = -float('inf')
        if battle.trapped:
            given_actions = battle.available_moves
        else:
            given_actions = battle.available_moves + battle.available_switches
        if len(given_actions) == 0:
            return self.choose_random_move(battle)
        for action in given_actions:
            action_vector = get_action_vector(action)
            model_input = self.get_model_input(gs, action_vector)
            with torch.no_grad():
                value = self.model(model_input)
            if value > best_value:
                best_action = action
                best_gs_action = model_input
                best_value = value
        self.gs_actions.append(best_gs_action)
        return self.create_order(best_action)


class PokeZeroTrain(PokeZero):
    def __init__(self, server_configuration, net, exploration=1, decay=1):
        super().__init__(server_configuration=server_configuration, net=net)
        self.predictions = Counter()
        self.exploration = exploration
        self.exploration_decay = decay

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.turn == 1:
            # print(f"exploration: {self.exploration}")
            self.exploration *= self.exploration_decay
        self.model.eval()
        gs = game_state(battle)  # 9828
        best_action = None
        best_gs_action = None
        best_value = -float('inf')
        given_actions = battle.available_moves + battle.available_switches
        if len(given_actions) == 0:
            return self.choose_random_move(battle)
        if random.random() < self.exploration:
            try:
                best_action = random.choice(given_actions)
            except IndexError:
                print(len(given_actions))
                print(given_actions)
                return self.choose_random_move(battle)
            best_gs_action = self.get_model_input(
                gs, get_action_vector(best_action))
            with torch.no_grad():
                best_value = self.model(best_gs_action)
        else:
            for action in given_actions:
                action_vector = get_action_vector(action)
                model_input = self.get_model_input(gs, action_vector)
                with torch.no_grad():
                    value = self.model(model_input)
                if value > best_value:
                    best_action = action
                    best_gs_action = model_input
                    best_value = value
        self.predictions[best_gs_action] = best_value
        return self.create_order(best_action)


class PokeZeroBattle(PokeZero):
    def __init__(self, server_configuration, net, player_configuration=None):
        super().__init__(server_configuration=server_configuration,
                         player_configuration=player_configuration, net=net)
        self.model.eval()

    @torch.no_grad()
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        gs = game_state(battle)  # 9828
        best_action = None
        best_value = -float('inf')
        given_actions = battle.available_moves + battle.available_switches
        if len(given_actions) == 0:
            return self.choose_random_move(battle)
        for action in given_actions:
            action_vector = get_action_vector(action)
            model_input = self.get_model_input(gs, action_vector)
            value = self.model(model_input.float())
            if value > best_value:
                best_action = action
                best_value = value
        return self.create_order(best_action)
