import numpy as np
from poke_env.environment.pokemon import Pokemon


def game_state(battle):
    """Returns vector representation of battle game state"""
    # todo: The vector should be the following: [current pokemon vector, all other pokemon in team vectors, opponent pokemon, all known opponent pokemon vectors]


def pokemon_name_vector(pokemon_names):
    """Returns a one-hot encoded vector corresponding to pokemon_names"""
    vector = []
    with open("random_battle_pool.txt", "r") as fin:
        for name in map(str.rstrip, fin):
            if name in pokemon_names:
                vector.append(1)
            else:
                vector.append(0)
    return np.array(vector)


def moves_vector(moves):
    """Returns a one-hot encoded vector corresponding to moves"""
    vector = []
    with open("random_battle_moveset.txt", "r") as fin:
        for move in map(str.rstrip, fin):
            if move in moves:
                vector.append(1)
            else:
                vector.append(0)
    return np.array(vector)


def status_vector(status):
    """Returns a one-hot encoded vector corresponding to status"""
    v = np.zeros(7)
    v[status - 1] = 1
    return v


def item_vector(item_name):
    """Returns a one-hot encoded vector corresponding to item_name"""
    v = []
    with open("random_battle_itemset.txt", "r") as fin:
        for item in map(str.rstrip, fin):
            if item == item_name:
                v.append(1)
            else:
                v.append(0)
    return np.array(v)


def abilities_vector(abilities):
    """Returns a one-hot encoded vector corresponding to abilities"""
    v = []
    with open("random_battle_abilityset.txt", "r") as fin:
        for ability in map(str.rstrip, fin):
            if ability in abilities:
                v.append(1)
            else:
                v.append(0)
    return np.array(v)


def base_stats_vector(bs: dict):
    n = 255
    return np.array([bs["hp"] / n, bs["atk"] / n, bs["def"] / n, bs["spa"] / n, bs["spd"] / n, bs["spe"] / n])


def boosts_vector(boosts: dict):
    n = 6
    return np.array(
        [boosts["accuracy"] / n, boosts["atk"] / n, boosts["def"] / n, boosts["evasion"] / n, boosts["spa"] / n,
         boosts["spd"] / n, boosts["spe"] / n])


def effects_vector(effect_names):
    """Returns a one-hot encoded vector corresponding to effect_names"""
    v = []
    with open("effectset.txt", "r") as fin:
        for effect in map(str.rstrip, fin):
            if effect in effect_names:
                v.append(1)
            else:
                v.append(0)
    return np.array(v)


def pokemon_vector(pokemon: Pokemon):
    """Given a pokemon object, return a vector representation"""
    # todo: [name, ability, active, base stats, boosts, current_hp_fraction, effects, fainted, first_turn, is_dynamaxed,
    #  item, level_fraction, moves, must_recharge, possible_abilities, preparing, protect_counter, status, status_counter,
    #  types,
    ability = abilities_vector([pokemon.ability])
    active = np.array([int(pokemon.active)])
    base_stats = base_stats_vector(pokemon.base_stats)
    boosts = boosts_vector(pokemon.boosts)
    hp = np.array([pokemon.current_hp_fraction])
    effects = effects_vector([e.name for e in pokemon.effects])
    fainted = np.array([int(pokemon.fainted)])
    #first_turn = np.array([int(pokemon.first_turn)]) # todo looks like it hasn't been added yet
    is_dynamaxed = np.array([int(pokemon.is_dynamaxed)])
    item = item_vector(pokemon.item)
    level = np.array([pokemon.level / 100])