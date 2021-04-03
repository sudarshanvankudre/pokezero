import pickle

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_dataset(dataset: int):
    """Loads dataset from disk and returns it in train test format"""
    with open("datasets/dataset{}.pickle".format(dataset), 'rb') as fin:
        dataset = pickle.load(fin)
        X, y = [], []
        for k, v in dataset.items():
            X.append(torch.squeeze(k).numpy())
            y.append(v)
        X, y = np.array(X), np.array(y)
    return X, y


def labels_array(X):
    """Returns an array of indices indicating which cluster the corresponding sample of X belongs to"""
    best_silhouette_score = -1
    best_n_clusters = 2
    for n_clusters in range(2, 101):
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        if silhouette_avg > best_silhouette_score:
            best_n_clusters = n_clusters
            best_silhouette_score = silhouette_avg
    return KMeans(n_clusters=best_n_clusters, random_state=10).fit_predict(X)
