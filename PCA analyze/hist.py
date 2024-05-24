# histogram for image R, G, B

import os, sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import numpy as np

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

image_good_directory = "../Data/train/good"
image_bad_directory = "../Data/train/bad"
# pixel :256 * 192 * 3


# Select a random 10 images from the good apple directory, bad apple directory
def select_random_images(directory):
    images = []
    for i in range(10):
        image = imread(os.path.join(directory, np.random.choice(os.listdir(directory))))
        images.append(image)
    return images


# histogram for images
def histogram_images(images, title):
    fig, axs = plt.subplots(1, 10, figsize=(20, 4))
    for i, image in enumerate(images):
        axs[i].hist(image[:, :, 0].flatten(), bins=256, color="red", alpha=0.5)
        axs[i].hist(image[:, :, 1].flatten(), bins=256, color="green", alpha=0.5)
        axs[i].hist(image[:, :, 2].flatten(), bins=256, color="blue", alpha=0.5)
        axs[i].set_title("Image " + str(i + 1))
    fig.suptitle(title)


histogram_images(select_random_images(image_good_directory), "Good Apple")
histogram_images(select_random_images(image_bad_directory), "Bad Apple")

plt.show()
