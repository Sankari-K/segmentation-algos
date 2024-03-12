import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

def create_collage(img_paths, output_path):
    """
    Creates a collage of 4 images in a 2x2 grid with labels. Resizes images to fit the collage while maintaining aspect ratios.
    Adds padding space between images and changes the font to black Arial 12pt.

    Args:
        img_paths: List of paths to the 4 images.
        output_path: Path to save the final collage image.
    """
    rows, cols = 2, 2
    padding = 10  # Padding space between images

    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

    for i, path in enumerate(img_paths):
        img = plt.imread(path)
        axs[i // cols, i % cols].imshow(img)
        axs[i // cols, i % cols].axis('off')

    for ax, label in zip(axs.flat, ["Original", "Benchmark Segmentation", "K-Means Clustering", "Mean Shift Clustering"]):
        ax.annotate(label, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=12, color='black', fontname='Arial')

    fig.tight_layout(pad=padding, h_pad=0.5)  # Adjust h_pad to reduce space between rows
    plt.savefig(output_path, bbox_inches='tight')

# # Example usage
# img_paths = ["datasets/resized/30.jpg", "datasets/segmented/30.jpg", "kmeans/output/cropped/30.jpg",
#              "meanshift/output/cropped/30.jpg"]
# output_path = "collage30.png"

# create_collage(img_paths, output_path)

# print("Collage created successfully!")

img1, img2, img3, img4 = [], [], [], []
for i in range(1,31):
    img1.append(f"datasets/resized/{i:02d}.jpg")
    img2.append(f"datasets/segmented/{i:02d}.jpg")
    img3.append(f"kmeans/output/cropped/{i:02d}.jpg")
    img4.append(f"meanshift/output/cropped/{i:02d}.jpg")

for i in range(1, 31):
    output_path = f"results/collage{i:02d}.png"
    create_collage([img1[i], img2[i], img3[i], img4[i]], output_path)
    print(f"Collage {i:02d} created successfully!")
