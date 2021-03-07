def random_battle_total_pokemon():
    """Returns the total number of pokemon in the random battle pool"""
    with open("random_battle_pool.txt", "r") as fin:
        return len(list(fin))


def random_battle_total_moves():
    """Returns the total number of moves in the random battle moveset"""
    with open("random_battle_moveset.txt", "r") as fin:
        return len(list(fin))
