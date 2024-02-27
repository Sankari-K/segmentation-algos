# imports
import numpy as np # for handling image data
from sklearn.datasets import make_blobs # to generate data points for some visualizations
from collections import defaultdict  # to store cluster information
import matplotlib.pyplot as plt  # for plotting
import random  # to initialize randomly generated clusters
import cv2 # converts image to pixel values

# select k centroids
def get_initial_centroids(X, k):
    """
    Function selects k random data points from X
    """
    initial_centroids = set(np.random.choice(np.arange(len(X)), size=k))
    while len(initial_centroids) != k:
        initial_centroids.add(random.randrange(0, len(X)))
    initial_centroids = [tuple(X[id]) for id in initial_centroids]
    return np.array(initial_centroids)

def get_euclidean_matrix(A_matrix, B_matrix):
    A_square = np.reshape(np.sum(A_matrix * A_matrix, axis=1), (A_matrix.shape[0], 1))
    B_square = np.reshape(np.sum(B_matrix * B_matrix, axis=1), (1, B_matrix.shape[0]))
    AB = A_matrix @ B_matrix.T

    C = -2 * AB + B_square + A_square
    return np.sqrt(abs(C))

def get_clusters(X, centroids):
    """
    Returns a mapping of clusters to points that belong to that cluster
    """
    clusters = defaultdict(list)
    distance_matrix = get_euclidean_matrix(X, centroids)
    closest_cluster_ids = np.argmin(distance_matrix, axis=1)
    for i, cluster_id in enumerate(closest_cluster_ids):
        clusters[cluster_id].append(X[i])
    return clusters

def check_clusters_converged(old_centroids, new_centroids, threshold):
    distances_between_old_and_new_centroids = get_euclidean_matrix(old_centroids, new_centroids)
    return np.max(distances_between_old_and_new_centroids.diagonal()) <= threshold

def k_means(X, k, threshold=0.5):
    new_centroids = get_initial_centroids(X, k)

    has_converged = False
    while not has_converged:
        previous_centroids = new_centroids
        previous_clusters = get_clusters(X, previous_centroids)
        
        new_centroids = np.array([np.mean(c, axis=0) for c in previous_clusters.values()])

        has_converged = check_clusters_converged(previous_centroids, new_centroids, threshold)
    return new_centroids

def visualization():
    k = 4
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=k)

    centroids = k_means(X, k)
    clusters = get_clusters(X, centroids)

    plt.rcParams['figure.figsize'] = [10, 5]
    for centroid, points in clusters.items():
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        plt.scatter(points[:, 0], points[:, 1], marker='o')
        plt.grid()
        plt.scatter(centroid[0], centroid[1], marker='x', color="red")

    plt.show()

def get_segmented_image(image, k):
    image = cv2.imread(image)
    height, width, depth = image.shape

    X = np.reshape(image, (height * width, depth))
    X = np.array(X, dtype=np.int32)
    
    centroids = k_means(X, k)
    distance_matrix = get_euclidean_matrix(X, centroids)
    closest_cluster_ids = np.argmin(distance_matrix, axis=1)

    X_segmented = centroids[closest_cluster_ids]
    X_segmented = np.array(X_segmented, dtype=np.uint8)

    segmented_image = np.reshape(X_segmented, (height, width, depth))
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.savefig('segmented_image.png')
    plt.show()
    

# visualization()
get_segmented_image("image.jpg", 15)



