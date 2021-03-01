from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player


class MaxDamagePlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)
