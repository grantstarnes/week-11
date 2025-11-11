# --------------- Importing Libraries ---------------
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns

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

# Loading the diamonds dataset from seaborn
diamonds = sns.load_dataset("diamonds")

# Only look at the numerical columns
diamonds_num_columns = diamonds.select_dtypes(include = ['number'])

def kmeans_diamonds(n, k):
    """
    This function kmeans_diamonds takes in n and k, and uses the kmeans function from exercise 1 to 
    cluster the first n rows of the numerical columns of the diamonds dataset into k clusters. It then 
    returns the centroids and labels as a tuple (centroids, labels).
    """
    num_set = diamonds_num_columns.head(n).to_numpy()
    centroids, labels = kmeans(num_set, k)
    return centroids, labels

# --------------- Exercise 3 ---------------
def kmeans_timer(n, k, n_iter = 5):
    return None