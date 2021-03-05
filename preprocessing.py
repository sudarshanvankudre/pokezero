import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.status import Status
from poke_env.environment.weather import Weather


def game_state(battle: AbstractBattle):
    """Returns vector representation of battle game state"""
    our_team = np.concatenate([pokemon_vector(p) for p in battle.team.values()])
    opponent_team = np.concatenate([pokemon_vector(p, friendly=False) for p in battle.opponent_team.values()])
    return np.concatenate((our_team, opponent_team))


def status_onehot_vector(status: Status):
    """Returns a one-hot encoded vector corresponding to status"""
    v = np.zeros(7)
    if not status:
        return v
    v[status.value - 1] = 1
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
            raw_ability = ability.lower().replace(' ', '')
            if ability in abilities or raw_ability in abilities:
                v.append(1)
            else:
                v.append(0)
    return np.array(v)


def base_stats_vector(bs: dict):
    n = 255
    return np.array([bs["hp"] / n, bs["atk"] / n, bs["def"] / n, bs["spa"] / n, bs["spd"] / n, bs["spe"] / n])


def boosts_vector(boosts: dict):
    n = 6
    if not boosts:
        return np.zeros(7)
    v = []
    for b in ["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"]:
        if b in boosts:
            v.append(boosts[b] / n)
        else:
            v.append(0)
    return np.array(v)


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


def pokemon_vector(pokemon: Pokemon, friendly=True):
    """Given a pokemon object, return a vector representation"""
    if not friendly:
        ability = abilities_onehot_vector([pokemon.possible_abilities])
    else:
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
    move_vectors = [move_vector(m) for m in pokemon.moves.values()]
    padding = np.zeros(75 - len(move_vectors))
    moves = np.concatenate((np.ravel(move_vectors), padding))
    must_recharge = np.array([int(pokemon.must_recharge)])
    possible_abilities = abilities_onehot_vector(list(pokemon.possible_abilities.values()))
    if type(pokemon.preparing) is tuple or pokemon.preparing:
        preparing = np.array([1])
    else:
        preparing = np.array([0])
    status = status_onehot_vector(pokemon.status)
    types = types_onehot_vector(pokemon.types)
    return np.concatenate((ability, active, base_stats, boosts, hp, effects, fainted, is_dynamaxed, item, level, moves,
                           must_recharge, possible_abilities, preparing, status, types))


def category_onehot_vector(category: MoveCategory):
    """Returns a one-hot vector representation of category"""
    v = np.zeros(3)
    if not category:
        return v
    v[category.value - 1] = 1
    return v


def types_onehot_vector(types: iter):
    """Returns a one-hot vector representation of type(s)"""
    v = np.zeros(18)
    if not types:
        return v
    for t in types:
        if t:
            v[t.value - 1] = 1
    return v


def weather_onehot_vector(weather: Weather):
    v = np.zeros(7)
    if not weather:
        return v
    v[weather.value - 1] = 1
    return v


def move_vector(move: Move):
    """Given a move object, return a vector representation"""
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
        [int(move.ignore_ability), int(move.ignore_defensive), int(move.ignore_evasion)])
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
    move_type = types_onehot_vector([move.type])
    use_target_offensive = np.array([int(move.use_target_offensive)])
    weather = weather_onehot_vector(move.weather)
    return np.concatenate((accuracy, power, boosts, breaks_protect, can_z_move, category, crit_ratio, pp,
                           defensive_category, drain, expected_hits, force_switch, heal, ignores, is_z, no_pp_boosts,
                           non_ghost_target, priority, recoil, self_boost, sleep_usable, stalling_move, status,
                           steals_boosts, thaws_target, move_type, use_target_offensive, weather))


@DeprecationWarning
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


@DeprecationWarning
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
