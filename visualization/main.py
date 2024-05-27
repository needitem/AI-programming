import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

# Append the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Load the model
model = tf.keras.models.load_model("./model.keras")
model.summary()

# Print the shape of kernels in convolutional layers
for layer in model.layers:
    if "conv" in layer.name:
        kernel, bias = layer.get_weights()
        print(f"{layer.name}: {kernel.shape}")

# Normalize the kernel weights of the first layer for visualization
kernel, bias = model.layers[0].get_weights()
minval, maxval = kernel.min(), kernel.max()
kernel = (kernel - minval) / (maxval - minval)
n_kernel = kernel.shape[3]

# Visualize the convolutional kernels
plt.figure(figsize=(10, 10))
plt.suptitle("Convolutional Kernels")

for i in range(n_kernel):
    f = kernel[:, :, :, i]
    for j in range(3):
        plt.subplot(3, n_kernel, j * n_kernel + i + 1)
        plt.imshow(f[:, :, j], cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(str(i) + " " + str(j))
plt.show()

# Print the output shapes of convolutional layers
for layer in model.layers:
    if "conv" in layer.name:
        print(layer.name, layer.output.shape)

from tensorflow.keras.models import Model

# Create a partial model to get the output of the first layer
partial_model = Model(inputs=model.inputs, outputs=model.layers[0].output)
partial_model.summary()

# Load and preprocess the image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims

img = load_img("Data/test/good/20190809_115448.png", target_size=(240, 240))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = img / 255.0

# Predict feature maps
feature_maps = partial_model.predict(img)
n_feature_maps = feature_maps.shape[-1]

# Visualize the feature maps
square = int(n_feature_maps**0.5)
if square * square < n_feature_maps:
    square += 1

plt.figure(figsize=(12, 12))
plt.suptitle("Feature Maps")
ix = 1

for _ in range(square):
    for _ in range(square):
        if ix > n_feature_maps:
            break
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[0, :, :, ix - 1], cmap="gray")
        ix += 1
plt.show()
