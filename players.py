import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from poke_env.environment.status import Status

from preprocessing import game_state, moves_onehot_vector, pokemon_name_onehot_vector

import random

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
        print(list(filter(lambda p: battle.team[p].status != Status.FNT, battle.team)))
        actions = np.concatenate((battle.available_moves, battle.available_switches))
        print(actions)
        print(len(actions))
        # for action in actions:
        #     if type(action) is Move:
        #         action_vector = moves_onehot_vector([action.id])
        #     elif type(action) is Pokemon:
        #         action_vector = pokemon_name_onehot_vector([action])
        return self.create_order(random.choice(actions))