import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from progress.bar import ChargingBar
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
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


def preclustering(X):
    """Returns projection of X onto a lower dimension space"""
    pca = PCA(n_components=10)
    return pca.fit_transform(X)


def labels_array(X):
    """Returns an array of indices indicating which cluster the corresponding sample of X belongs to"""
    best_silhouette_score = -1
    best_cluster_labels = None
    max_n_clusters = 500
    min_n_clusters = 100
    bar = ChargingBar("Finding optimal k", max=max_n_clusters - 1)
    silhouette_scores = []
    for n_clusters in range(min_n_clusters, max_n_clusters + 1):
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=10, compute_labels=True)
        clusterer.fit(X)
        cluster_labels = clusterer.labels_
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_cluster_labels = cluster_labels
        bar.next()
    bar.finish()
    plt.title("Silhouette Score vs. Num Clusters")
    plt.xlabel("Num Clusters")
    plt.ylabel("Silhouette Score")
    plt.plot(np.arange(min_n_clusters, max_n_clusters + 1), silhouette_scores)
    plt.show()
    return best_cluster_labels


def win_rates(labels_array, y):
    """Returns an array of win rates for each input in X"""
    unique, counts = np.unique(labels_array, return_counts=True)
    victories = np.zeros(len(unique))
    for i in range(len(y)):
        if y[i] == 1:
            victories[labels_array[i]] += 1
    cluster_win_rate = np.array([victories[i] / counts[i] for i in range(len(victories))])
    win_rates = np.array([cluster_win_rate[i] for i in labels_array])
    return win_rates


def get_model_training_data(X, y):
    trans_X = preclustering(X)
    labels = labels_array(trans_X)
    win_rate = win_rates(labels, y)
    return X, win_rate
