import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import matplotlib.pyplot as plt
import random
from PIL import Image
from numpy import asarray

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
good_apple = next(os.walk("Data/Apple_Good"))[2]
bad_apple = next(os.walk("Data/Apple_Bad"))[2]


def split_files(src_dir, ratio, test_dir, train_dir):
    # Get a list of all files in the source directory
    files = os.listdir(src_dir)
    count = int(len(files) * ratio)
    # Shuffle the list of files
    random.shuffle(files)

    for i in range(count):
        # Move the first `count` files to the test directory
        os.rename(os.path.join(src_dir, files[i]), os.path.join(test_dir, files[i]))

    for i in files[count:]:
        # Move the rest of the files to the train directory
        os.rename(os.path.join(src_dir, i), os.path.join(train_dir, i))


# pca for good_apple and bad_apple


labels = ["Good", "Bad"]

test_dir_good = "Data/test/good"
train_dir_good = "Data/train/good"
test_dir_bad = "Data/test/bad"
train_dir_bad = "Data/train/bad"

if not os.path.exists(test_dir_good):
    os.makedirs(test_dir_good)
if not os.path.exists(train_dir_good):
    os.makedirs(train_dir_good)
if not os.path.exists(test_dir_bad):
    os.makedirs(test_dir_bad)
if not os.path.exists(train_dir_bad):
    os.makedirs(train_dir_bad)

split_files("Data/Apple_Good", 0.2, test_dir_good, train_dir_good)
split_files("Data/Apple_Bad", 0.2, test_dir_bad, train_dir_bad)

datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.1)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_data = datagen.flow_from_directory(
    train_path="Data/train",
    shuffle=True,
    target_size=(240, 240),
    classes=labels,
    batch_size=32,
    class_mode="categorical",
    subset="training",
)

val_data = datagen.flow_from_directory(
    train_path="Data/train",
    shuffle=True,
    target_size=(240, 240),
    classes=labels,
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 3)))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="softmax"))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    validation_data=val_data,
    validation_steps=len(val_data),
    epochs=10,
    verbose=1,
)
result = model.evaluate(train_data, verbose=1)
print("accuracy: ", result[1] * 100)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "validate"], loc="upper left")
plt.grid()
plt.show()
