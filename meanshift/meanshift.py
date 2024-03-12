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
        if (i+1)%1000==0: print(f"Processed {i+1}/{len(data)} points")
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

def get_segmented_image(image, bandwidth=5,output_path='segmented_image.png'):
    image = cv2.imread(image)
    height, width, depth = image.shape

    X = np.reshape(image, (height * width, depth)).astype(np.float64)

    shifted_points = mean_shift(X, bandwidth=bandwidth)

    segmented_image = np.reshape(shifted_points.astype(np.uint8), (height, width, depth))
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.savefig(output_path)
    # plt.show()

import os
for image in os.listdir('datasets/resized'):
    path = 'datasets/resized/' + image
    if image in os.listdir('meanshift/output/original'): 
        print("Already done with", image)
        continue #['01.jpg', '02.jpg', '03.jpg', '04.jpg', '05.jpg', '06.jpg', '07.jpg', '08.jpg', '09.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg']
    print(path)
    print("Processing", image)
    get_segmented_image(path, bandwidth=15, output_path=f'meanshift/output/original/{image}')
    print("Done with", image)
