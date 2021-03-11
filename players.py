import random
from collections import Counter

import numpy as np
import torch
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from preprocessing import game_state, move_name_onehot_vector, pokemon_species_onehot_vector
from stats import random_battle_total_pokemon, random_battle_total_moves

num_pokemon = random_battle_total_pokemon()
num_moves = random_battle_total_moves()


class MaxDamagePlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        pokemon = battle.active_pokemon
        for k, v in pokemon.moves.items():
            move = v
            print(move)
            print(move.terrain)
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


def get_model_input(gs, action_vector):
    action_vector_max_size = 11997
    model_input = np.concatenate((gs, action_vector))
    model_input = np.pad(model_input, (0, action_vector_max_size - model_input.shape[0]), 'constant',
                         constant_values=0)
    model_input = torch.unsqueeze(torch.unsqueeze(
        torch.from_numpy(model_input), 0), 0).float()
    # model_input = torch.from_numpy(model_input)
    return model_input.float()


class PokeZero(Player):
    def __init__(self, server_configuration, net, exploration=0.3):
        super().__init__(server_configuration=server_configuration)
        self.predictions = Counter()
        self.model = net
        self.random_move_chance = exploration

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        gs = game_state(battle)  # 9828
        best_action = None
        best_gs_action = None
        best_value = -float('inf')
        given_actions = battle.available_moves + battle.available_switches
        if random.random() < self.random_move_chance:
            try:
                best_action = random.choice(given_actions)
            except IndexError:
                print(len(given_actions))
                print(given_actions)
                return self.choose_random_move(battle)
            best_gs_action = get_model_input(
                gs, get_action_vector(best_action))
            best_value = self.model(best_gs_action)
        else:
            for action in given_actions:
                action_vector = get_action_vector(action)
                model_input = get_model_input(gs, action_vector)
                value = self.model(model_input)
                if value > best_value:
                    best_action = action
                    best_gs_action = model_input
                    best_value = value
        self.predictions[best_gs_action] = best_value
        return self.create_order(best_action)


class PokeZeroField(Player):
    def __init__(self, server_configuration, net, player_configuration=None):
        super().__init__(server_configuration=server_configuration,
                         player_configuration=player_configuration)
        self.model = net

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        gs = game_state(battle)  # 9828
        best_action = None
        best_value = -float('inf')
        given_actions = battle.available_moves + battle.available_switches
        for action in given_actions:
            action_vector = get_action_vector(action)
            model_input = get_model_input(gs, action_vector)
            value = self.model(model_input.float())
            if value > best_value:
                best_action = action
                best_value = value
        return self.create_order(best_action)
