import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


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
    pca = PCA(n_components=100)
    return pca.fit_transform(X)


def labels_array(X):
    """Returns an array of indices indicating which cluster the corresponding sample of X belongs to"""
    best_silhouette_score = -1
    best_cluster_labels = None
    max_n_clusters = 200
    min_n_clusters = 100
    silhouette_scores = []
    for n_clusters in range(min_n_clusters, max_n_clusters + 1, 50):
        # print("Num of clusters: {}".format(n_clusters))
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=10, compute_labels=True)
        clusterer.fit(X)
        cluster_labels = clusterer.labels_
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_cluster_labels = cluster_labels
    plt.title("Num Dimensions: {}".format(X.shape[1]))
    plt.xlabel("Num Clusters")
    plt.ylabel("Silhouette Score")
    plt.plot(np.arange(min_n_clusters, max_n_clusters + 1, 50), silhouette_scores)
    plt.savefig("clustering_evals/clustering_eval.png")
    return best_cluster_labels


def win_rates(labels_array, y):
    """Returns an array of win rate for each input in X"""
    counts = Counter()
    victories = Counter()
    for i in range(len(y)):
        if y[i] == 1:
            victories[labels_array[i]] += 1
        counts[labels_array[i]] += 1
    return np.array([victories[cluster] / counts[cluster] for cluster in labels_array])


def get_model_training_data(X, y):
    trans_X = preclustering(X)
    labels = labels_array(trans_X)
    win_rate = win_rates(labels, y)
    return X, win_rate
