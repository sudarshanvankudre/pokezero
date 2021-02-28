import re

import requests

gen8_random_battle_url = "https://pkmn.github.io/randbats/data/gen8randombattle.js?"


def load_random_battle_pool():
    r = requests.get(gen8_random_battle_url)
    random_battle_pool = [s[:len(s) - 3] for s in filter(lambda s: s[0].isupper(), re.findall(r"[^ ]*?: {", r.text))]
    with open("random_battle_pool.txt", 'w') as fout:
        for pokemon in random_battle_pool:
            fout.write(pokemon + "\n")


def load_random_battle_moveset():
    r = requests.get(gen8_random_battle_url)
    moves = set()
    for move_list in map(lambda s: eval(s[7:]), re.findall(r"moves: \[.*?\]", r.text)):
        for m in move_list:
            moves.add(m)
    with open("random_battle_moveset.txt", 'w') as fout:
        for move in sorted(moves):
            fout.write(move + "\n")


load_random_battle_moveset()
