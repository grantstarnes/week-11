# ------------------------- Importing Libraries -------------------------
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from time import time

# ------------------------- Exercise 1 -------------------------

def kmeans(X, k):
    """
    This kmeans function takes in a numpy array X and an integer k and returns the centroids and labels after fitting the KMeans model.
    It utilizes KMeans from sklearn with a fixed random state for reproducibility, 42, since it is best practice.
    It then returns the centroids and labels as a tuple (centroids, labels).
    """

    # Creating the KMeans model with k clusters and random state 42
    cluster_model = KMeans(n_clusters = k, random_state = 42)

    # Fitting the model to the data X
    cluster_model.fit(X)

    # Getting the centroids
    centroids = cluster_model.cluster_centers_

    # Getting the labels
    labels = cluster_model.labels_

    # Returns the centroids and labels as a tuple (centroids, labels)
    return centroids, labels

# ------------------------- Exercise 2 -------------------------

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
    # Selecting the first n rows of the numerical columns of the diamonds dataset
    num_set = diamonds_num_columns.head(n).to_numpy()

    # Using the kmeans function to cluster the selected data into k clusters
    centroids, labels = kmeans(num_set, k)

    # Returns the centroids and labels as a tuple (centroids, labels)
    return centroids, labels

# ------------------------- Exercise 3 -------------------------

def kmeans_timer(n, k, n_iter = 5):
    """
    This function kmeans_timer takes in n, k, and n_iter (5 iterations by default) and runs the kmeans_diamonds function n_iter times,
    recording the time taken for each run. It returns the average time taken over the 5 n_iter runs.
    """
    # Empty list to store the 5 time measurements
    times = []

    # Running kmeans_diamonds n_iter times and recording the time taken for each run using a for loop
    for _ in range(n_iter):
        start_time = time()
        _ = kmeans_diamonds(n, k)
        time_elapsed = time() - start_time
        times.append(time_elapsed)
    
    # Calculate the average time taken for the 5 n_iter runs
    average_time = np.mean(times)

    # Returns the average time
    return average_time