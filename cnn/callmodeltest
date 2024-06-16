import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Append the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Load the model
model = tf.keras.models.load_model("./resnet.keras")
model.summary()

labels = ["good", "bad"]

train_path = "Data/train"
test_path = "Data/test"

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    classes=labels,
    target_size=(240, 240),
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

# Evaluate the Model
loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Get Predictions and True Labels
filenames = test_generator.filenames
nb_samples = len(filenames)
predict = model.predict(test_generator, steps=nb_samples)
y_pred = np.argmax(predict, axis=-1)

y_true = test_generator.classes

# Display a Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)

# Visualization (Optional)
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Detailed Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

# Misclassified Images (Optional)
misclassified_indices = np.where(y_pred != y_true)[0]
if misclassified_indices.size > 0:
    print("\nMisclassified Images:")
    plt.figure(figsize=(12, 8))
    for i in range(min(9, misclassified_indices.size)):  # Show a maximum of 9 images
        plt.subplot(3, 3, i + 1)
        index = misclassified_indices[i]
        img = plt.imread(os.path.join(test_path, filenames[index]))
        plt.imshow(img)
        plt.title(f"True: {labels[y_true[index]]}, Pred: {labels[y_pred[index]]}")
        plt.axis("off")
    plt.show()
