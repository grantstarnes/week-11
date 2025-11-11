# --------------- Importing Libraries ---------------
import numpy as np
from sklearn.cluster import KMeans

# --------------- Exercise 1 ---------------
def kmeans(X, k):
    """
    This kmeans function takes in a numpy array X and an integer k and returns the centroids and labels after fitting the KMeans model.
    It utilizes KMeans from sklearn with a fixed random state for reproducibility, 42, since it is best practice.
    It then returns the centroids and labels as a tuple (centroids, labels).
    """
    cluster_model = KMeans(n_clusters = k, random_state = 42)

    cluster_model.fit(X)

    centroids = cluster_model.cluster_centers_
    labels = cluster_model.labels_

    return centroids, labels

# --------------- Exercise 2 ---------------
def kmeans_diamonds(n, k):
    return None

# --------------- Exercise 3 ---------------
def kmeans_timer(n, k, n_iter = 5):
    return None