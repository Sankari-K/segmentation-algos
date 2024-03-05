import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import cv2

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth))**2)

def get_new_centroid(x, data, bandwidth):
    weights = gaussian_kernel(np.linalg.norm(data - x, axis=1), bandwidth)
    return np.sum(data * weights[:, np.newaxis], axis=0) / np.sum(weights)

def mean_shift(data, bandwidth=2, convergence_threshold=0.001, max_iterations=100):
    shifted_points = data.copy()
    for i, point in enumerate(data):
        old_point = point
        shift = np.inf
        iteration = 0
        while shift > convergence_threshold and iteration < max_iterations:
            new_point = get_new_centroid(old_point, data, bandwidth)
            shift = euclidean_distance(new_point, old_point)
            old_point = new_point
            iteration += 1
        shifted_points[i] = old_point
        print(f"Processed {i+1}/{len(data)} points")
    return shifted_points

def visualization():
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=4)

    shifted_points = mean_shift(X)

    plt.rcParams['figure.figsize'] = [10, 5]
    plt.scatter(X[:, 0], X[:, 1], marker='o', label='Original Data')
    plt.scatter(shifted_points[:, 0], shifted_points[:, 1], marker='x', color="red", label='Cluster Centers')
    plt.legend()
    plt.grid()
    plt.show()

def get_segmented_image(image, bandwidth=5):
    image = cv2.imread(image)
    height, width, depth = image.shape

    X = np.reshape(image, (height * width, depth)).astype(np.float64)

    shifted_points = mean_shift(X, bandwidth=bandwidth)

    segmented_image = np.reshape(shifted_points.astype(np.uint8), (height, width, depth))
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.savefig('segmented_image1.png')
    plt.show()

get_segmented_image("../dataset/01.jpg", bandwidth=25)
