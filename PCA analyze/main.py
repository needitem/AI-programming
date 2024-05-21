import os, sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import numpy as np

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

image_raw = imread("../Data/train/bad/IMG_20190824_175205.png")
# pixel :256 * 192 * 3
print(image_raw.shape)

plt.figure(figsize=[12, 8])
plt.imshow(image_raw)

image_sum = image_raw.sum(axis=2)
print(image_sum.shape)

image_bw = image_sum / image_sum.max()
print(image_bw.max())

plt.figure(figsize=[12, 8])
plt.imshow(image_bw, cmap="gray")

from sklearn.decomposition import PCA, IncrementalPCA

pca = PCA()
pca.fit(image_bw)

# Getting the cumulative variance

var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100

# How many PCs explain 95% of the variance?
k = np.argmax(var_cumu > 95)
print("Number of components explaining 95% variance: " + str(k))
# print("\n")

plt.figure(figsize=[10, 5])
plt.title("Cumulative Explained Variance explained by the components")
plt.ylabel("Cumulative Explained variance")
plt.xlabel("Principal components")
plt.axvline(x=k, color="k", linestyle="--")
plt.axhline(y=95, color="r", linestyle="--")
ax = plt.plot(var_cumu)

ipca = IncrementalPCA(n_components=k)
image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))

# Plotting the reconstructed image
plt.figure(figsize=[12, 8])
plt.imshow(image_recon, cmap=plt.cm.gray)
plt.show()
