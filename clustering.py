import pickle
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from multiprocessing import Pool


def load_dataset(dataset: int) -> dict:
    """Loads dataset from disk and returns it in train test format"""
    with open("datasets/dataset{}.pickle".format(dataset), 'rb') as fin:
        dataset = pickle.load(fin)
    return dataset


def transform_dataset(dataset: dict):
    X, y = [], []
    for k, v in dataset.items():
        X.append(torch.squeeze(k).numpy())
        y.append(v)
    return np.array(X), np.array(y)


def preclustering(X):
    """Returns projection of X onto a lower dimension space"""
    return PCA(n_components=100).fit_transform(X)


def labels_array(X):
    """Returns an array of indices indicating which cluster the corresponding sample of X belongs to"""
    def num_clusters_test(num):
        clusterer = MiniBatchKMeans(
            n_clusters=num, random_state=10, compute_labels=True)
        clusterer.fit(X)
        cluster_labels = clusterer.labels_
        return silhouette_score(X, cluster_labels), cluster_labels

    min_clusters = 500
    max_clusters = 1200

    with Pool() as p:
        results = p.map(num_clusters_test, range(
            min_clusters, max_clusters, 10))

    return max(results, key=lambda p: p[0])[1]


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
