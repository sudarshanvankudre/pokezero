import matplotlib.pyplot as plt
import numpy as np


def graph_results(opponent: str):
    with open(f"vs_{opponent}_results.txt", "r") as fin:
        results = list(map(float, fin))
        print(len(results))
        plt.xlabel("win rate")
        plt.ylabel("frequency")
        plt.title(f"{opponent} player")
        plt.hist(results)
        plt.show()
        plt.title(f"{opponent} player")
        plt.xlabel("# 20 games")
        plt.ylabel("win rate")
        x = np.arange(len(results))
        y = results
        plt.plot(x, y, 'o')
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * x + b)
        plt.show()


graph_results("random")
graph_results("max_damage")
