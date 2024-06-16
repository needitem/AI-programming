import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Append the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Load the model
model = tf.keras.models.load_model("./resnet.keras")
model.summary()

# Print the shape of kernels in convolutional layers (refined)
print("\nConvolutional Layer Kernel Shapes:")
for layer in model.layers:
    if "conv" in layer.name:
        weights = layer.get_weights()  # Get all weights (could be 0, 1, or 2)
        if weights:  # Check if weights exist
            kernel = weights[0]  # Extract kernel weights
            print(f"{layer.name}: {kernel.shape}")

# Normalize the kernel weights of the first convolutional layer for visualization
first_conv_layer = None
for layer in model.layers:
    if "conv" in layer.name and layer.get_weights():
        first_conv_layer = layer
        break

if first_conv_layer:
    kernel, bias = first_conv_layer.get_weights()
    minval, maxval = kernel.min(), kernel.max()
    kernel = (kernel - minval) / (maxval - minval)
    n_kernel = kernel.shape[3]

    # Visualize the convolutional kernels
    plt.figure(figsize=(10, 10))
    plt.suptitle("Convolutional Kernels (First Layer)")
    for i in range(n_kernel):
        f = kernel[:, :, :, i]
        for j in range(3):
            plt.subplot(3, n_kernel, j * n_kernel + i + 1)
            plt.imshow(f[:, :, j], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.title(str(i) + " " + str(j))
    plt.show()
else:
    print("No convolutional layers with weights found in the model.")

# Load and preprocess the image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims

img_path = "Data/test/good/20190809_115448.png"  # Replace with your image path
img = load_img(img_path, target_size=(240, 240))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = img / 255.0

# Visualize feature maps for all convolutional layers (refined)
print("\nFeature Maps:")
for layer_index, layer in enumerate(model.layers):
    if "conv" in layer.name:
        partial_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
        feature_maps = partial_model.predict(img)

        # Determine grid layout for visualization
        n_feature_maps = feature_maps.shape[-1]
        square = int(np.ceil(np.sqrt(n_feature_maps)))

        # Visualize feature maps
        plt.figure(figsize=(12, 12))
        plt.suptitle(f"Feature Maps - Layer {layer_index+1} ({layer.name})")
        for i in range(n_feature_maps):
            ax = plt.subplot(square, square, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(
                feature_maps[0, :, :, i], cmap="gray"
            )  # Assuming batch size of 1
        plt.show()
