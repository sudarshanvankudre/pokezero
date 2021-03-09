import random

import numpy as np
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
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


class MyRandomPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        gs = game_state(battle)
        return self.choose_random_move(battle)


class PokeZero(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # max action vector size is 10197
        gs = game_state(battle) # 9828
        best_action = None
        best_value = -float('inf')
        given_actions = battle.available_moves + battle.available_switches
        for action in given_actions:
            if type(action) is Move:
                action_vector = move_name_onehot_vector(action.id)
            else:
                action_vector = pokemon_species_onehot_vector(action.species)
            model_input = np.concatenate((gs, action_vector))
            model_input = np.pad(model_input, (0, 10197 - action_vector.shape[0]), 'constant', constant_values=0)
            value = model(model_input) # todo: build model
            if value > best_value:
                best_action = action
                best_value = value
        # return self.create_order(best_action)
        return self.choose_random_move(battle)
