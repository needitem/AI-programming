import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Append the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Define the labels
labels = ["good", "bad"]

# Define the paths for training and test data
train_path = "Data/train"
test_path = "Data/test"

# Data augmentation and preprocessing for training
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
)

test_datgen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load the training data
train_data = datagen.flow_from_directory(
    directory=train_path,
    shuffle=True,
    classes=labels,
    target_size=(240, 240),
    subset="training",
    class_mode="categorical",
    batch_size=32,
)

# Load the validation data
val_data = datagen.flow_from_directory(
    directory=train_path,
    shuffle=True,
    classes=labels,
    target_size=(240, 240),
    subset="validation",
    class_mode="categorical",
    batch_size=32,
)

# Build the CNN model
model = Sequential(
    [
        Input(shape=(240, 240, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Dropout(0.4),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Dropout(0.4),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
)

# Evaluate the model on the training data
result = model.evaluate(train_data, verbose=1)
print("Training accuracy: ", result[1] * 100)

# Plot training and validation accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validate"], loc="upper left")
plt.grid()
plt.show()

# Load and evaluate the test data
test_generator = test_datgen.flow_from_directory(
    directory=test_path,
    classes=labels,
    target_size=(240, 240),
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

loss, accuracy = model.evaluate(test_generator, verbose=1)
print("Test accuracy = %f; loss = %f" % (accuracy, loss))
print("Test loss: ", loss)

# Save the model
model.save("model.keras")
