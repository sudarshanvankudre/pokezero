from collections import Counter
from multiprocessing import Process, Queue, cpu_count

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def transform_dataset(dataset: dict):
    X, y = [], []
    for k, v in dataset.items():
        X.append(torch.squeeze(k).numpy())
        y.append(v)
    return np.array(X), np.array(y)


def preclustering(X):
    """Returns projection of X onto a lower dimension space"""
    return PCA(n_components=100).fit_transform(X)


def split_range(low, high, num_buckets):
    """Returns buckets ranges, splitting [low, high]
    """
    ranges = []
    div = (high - low) // num_buckets
    for i in range(num_buckets):
        if i == num_buckets - 1:
            ranges.append((i * div + low, high + 1))
        else:
            ranges.append((i * div + low, div * (i + 1) + low))
    return ranges


def labels_array(X):
    """Returns an array of indices indicating which cluster the corresponding sample of X belongs to"""

    def num_clusters_search(r: tuple, q: Queue):
        best_score = -1
        best_labels = None
        for num_clusters in range(r[0], r[1], 15):
            clusterer = MiniBatchKMeans(
                n_clusters=num_clusters, compute_labels=True)
            clusterer.fit(X)
            score = silhouette_score(X, clusterer.labels_)
            if score > best_score:
                best_labels = clusterer.labels_
                best_score = score
        q.put((best_score, best_labels))

    min_clusters = 500
    max_clusters = 1200
    ranges = split_range(min_clusters, max_clusters, cpu_count())
    q = Queue()
    processes = []
    for r in ranges:
        p = Process(target=num_clusters_search, args=(r, q))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    q.put("DONE")
    best_score = -1
    best_labels = None
    while (element := q.get()) != "DONE":
        score = element[0]
        if score > best_score:
            best_labels = element[1]
            best_score = score
    print(f"sillouette score: {best_score}")
    return best_labels


def win_rates(cluster_labels, y):
    """Returns an array of win rate for each input in X"""
    counts = Counter()
    victories = Counter()
    for i in range(len(y)):
        if y[i] == 1:
            victories[cluster_labels[i]] += 1
        counts[cluster_labels[i]] += 1
    return np.array([victories[cluster] / counts[cluster] for cluster in cluster_labels])


def get_model_training_data(X, y):
    return X, win_rates(labels_array(preclustering(X)), y)
