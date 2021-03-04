import numpy as np
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType


def game_state(battle):
    """Returns vector representation of battle game state"""
    # todo: The vector should be the following: [current pokemon vector, all other pokemon in team vectors, opponent pokemon, all known opponent pokemon vectors]


def pokemon_name_onehot_vector(pokemon_names):
    """Returns a one-hot encoded vector corresponding to pokemon_names"""
    vector = []
    with open("random_battle_pool.txt", "r") as fin:
        for name in map(str.rstrip, fin):
            if name in pokemon_names:
                vector.append(1)
            else:
                vector.append(0)
    return np.array(vector)


def moves_onehot_vector(moves):
    """Returns a one-hot encoded vector corresponding to moves"""
    vector = []
    with open("random_battle_moveset.txt", "r") as fin:
        for move in map(str.rstrip, fin):
            if move in moves:
                vector.append(1)
            else:
                vector.append(0)
    return np.array(vector)


def status_onehot_vector(status):
    """Returns a one-hot encoded vector corresponding to status"""
    v = np.zeros(7)
    v[status - 1] = 1
    return v


def item_onehot_vector(item_name):
    """Returns a one-hot encoded vector corresponding to item_name"""
    v = []
    with open("random_battle_itemset.txt", "r") as fin:
        for item in map(str.rstrip, fin):
            if item == item_name:
                v.append(1)
            else:
                v.append(0)
    return np.array(v)


def abilities_onehot_vector(abilities):
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


def effects_onehot_vector(effect_names):
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
    ability = abilities_onehot_vector([pokemon.ability])
    active = np.array([int(pokemon.active)])
    base_stats = base_stats_vector(pokemon.base_stats)
    boosts = boosts_vector(pokemon.boosts)
    hp = np.array([pokemon.current_hp_fraction])
    effects = effects_onehot_vector([e.name for e in pokemon.effects])
    fainted = np.array([int(pokemon.fainted)])
    # first_turn = np.array([int(pokemon.first_turn)]) # todo looks like it hasn't been added yet
    is_dynamaxed = np.array([int(pokemon.is_dynamaxed)])
    item = item_onehot_vector(pokemon.item)
    level = np.array([pokemon.level / 100])


def category_onehot_vector(category: MoveCategory):
    """Returns a one-hot vector representation of category"""
    if category == MoveCategory.PHYSICAL:
        return np.array([1, 0, 0])
    elif category == MoveCategory.SPECIAL:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


def type_onehot_vector(type: PokemonType):
    """Returns a one-hot vector representation of type"""
    types = [PokemonType.BUG, PokemonType.DARK, PokemonType.DRAGON, PokemonType.ELECTRIC, PokemonType.FAIRY]


def move_vector(move: Move):
    """Given a move object, return a vector representation"""
    # todo: [accuracy, base_power, boosts, breaks_protect, can_z_move, category, crit_ratio, current_pp, damage,
    #   defensive category, drain, expected_hits, force_switch, heal, ignore_ability, ignore_defensive, ignore_evasion,
    #   ignore_immunity, is_protect_counter, is_protect_move, is_side_protect_move, is_z, no_pp_boosts, non_ghost_target,
    #   priority, recoil, self_boost, sleep_usable, stalling_move, status, steals_boosts, terrain, thaws_target, type,
    #   use_target_offensive, weather
    accuracy = np.array([move.accuracy])
    power = np.array([move.base_power / 250])
    boosts = boosts_vector(move.boosts)
    breaks_protect = np.array([int(move.breaks_protect)])
    can_z_move = np.array([int(move.can_z_move)])
    category = category_onehot_vector(move.category)
    crit_ratio = np.array([move.crit_ratio / 6])
    pp = np.array([move.current_pp / move.max_pp])
    defensive_category = category_onehot_vector(move.defensive_category)
    drain = np.array([move.drain])
    expected_hits = np.array([move.expected_hits])
    force_switch = np.array([int(move.force_switch)])
    heal = np.array([move.heal])
    ignores = np.array(
        [int(move.ignore_ability), int(move.ignore_defensive), int(move.ignore_evasion), int(move.ignore_immunity)])
    is_z = np.array([int(move.is_z)])
    no_pp_boosts = np.array([int(move.no_pp_boosts)])
    non_ghost_target = np.array([int(move.non_ghost_target)])
    priority = np.array([move.priority / 5])
    recoil = np.array([move.recoil])
    self_boost = boosts_vector(move.self_boost)
    sleep_usable = np.array([int(move.sleep_usable)])
    stalling_move = np.array([int(move.stalling_move)])
    status = status_onehot_vector(move.status)
    steals_boosts = np.array([int(move.steals_boosts)])
    thaws_target = np.array([int(move.thaws_target)])
    type = type_onehot_vector(move.type)
