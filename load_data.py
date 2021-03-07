import re

import requests
from bs4 import BeautifulSoup

gen8_random_battle_url = "https://pkmn.github.io/randbats/data/gen8randombattle.js?"
battle_environment_url = "https://poke-env.readthedocs.io/en/stable/other_environment.html#poke_env.environment.effect.Effect"


def load_random_battle_pool():
    r = requests.get(gen8_random_battle_url)
    random_battle_pool = [s[:len(s) - 3] for s in filter(lambda s: s[0].isupper(), re.findall(r"[^ ]*?: {", r.text))]
    with open("random_battle_pool.txt", 'w') as fout:
        for pokemon in random_battle_pool:
            fout.write(pokemon + "\n")


def load_random_battle_moveset():
    r = requests.get(gen8_random_battle_url)
    moves = set()
    for move_list in map(lambda s: eval(s[7:]), re.findall(r"moves: \[.*?]", r.text)):
        for m in move_list:
            moves.add(m)
    with open("random_battle_moveset.txt", 'w') as fout:
        for move in sorted(moves):
            fout.write(move + "\n")


def load_random_battle_itemset():
    r = requests.get(gen8_random_battle_url)
    items = set()
    for item_list in map(lambda s: eval(s[7:]), re.findall(r"items: \[.*?]", r.text)):
        for item in item_list:
            items.add(item)
    with open("random_battle_itemset.txt", "w") as fout:
        for item in sorted(items):
            fout.write(item + "\n")


def load_random_battle_abilityset():
    r = requests.get(gen8_random_battle_url)
    abilities = set()
    for ability_list in map(lambda s: eval(s[11:]), re.findall(r"abilities: \[.*?]", r.text)):
        for ability in ability_list:
            abilities.add(ability)
    with open("random_battle_abilityset.txt", "w") as fout:
        for ability in sorted(abilities):
            fout.write(ability + "\n")


def load_effects():
    r = requests.get(battle_environment_url)
    soup = BeautifulSoup(r.text, 'html.parser').find("div",
                                                     {"class": "section", "id": "module-poke_env.environment.effect"})
    effects = [str(s)[23:len(s) - 8] for s in soup.find_all("code", {"class": "descname"})]
    with open("effectset.txt", "w") as fout:
        for effect in sorted(effects[1: len(effects) - 4]):
            fout.write(effect + "\n")


load_random_battle_pool()
load_effects()
load_random_battle_abilityset()
load_random_battle_itemset()
load_random_battle_moveset()
