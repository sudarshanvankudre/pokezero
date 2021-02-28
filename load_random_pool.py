import re

import requests

gen8_random_battle_url = "https://pkmn.github.io/randbats/data/gen8randombattle.js?"

r = requests.get(gen8_random_battle_url)
random_battle_pool = [s[:len(s) - 3] for s in filter(lambda s: s[0].isupper(), re.findall(r"[^ ]*?: {", r.text))]
with open("random_battle_pool.txt", 'w') as fout:
    for pokemon in random_battle_pool:
        fout.write(pokemon + "\n")
