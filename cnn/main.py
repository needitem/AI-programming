import os
import sys
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


labels = ["good", "bad"]

train_path = "Data/train"
test_path = "Data/test"

datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.1)
test_datgen = ImageDataGenerator(rescale=1.0 / 255.0)

train_data = datagen.flow_from_directory(
    directory=train_path,
    shuffle=True,
    classes=labels,
    target_size=(240, 240),
    subset="training",
    class_mode="categorical",
    batch_size=32,
)

val_data = datagen.flow_from_directory(
    directory=train_path,
    shuffle=True,
    classes=labels,
    target_size=(240, 240),
    subset="validation",
    class_mode="categorical",
    batch_size=32,
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(240, 240, 3)))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# model.summary()

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
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


test_generator = test_datgen.flow_from_directory(
    directory=test_path,
    classes=labels,
    target_size=(240, 240),
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

loss, accuracy = model.evaluate(test_generator, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
print("Loss: ", loss)
model.save("model.keras")
