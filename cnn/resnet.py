import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 레이블 정의
labels = ["good", "bad"]

# 훈련 및 테스트 데이터 경로
train_path = "Data/train"
test_path = "Data/test"

# --- Image Augmentation and Preprocessing ---

# Stronger Data Augmentation for Training (More Diversity)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,  # Increased rotation
    width_shift_range=0.25,  # Increased shift
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.3,  # Increased zoom
    horizontal_flip=True,
    vertical_flip=True,  # Added vertical flip
    fill_mode="nearest",
    validation_split=0.2,  # More validation data (20%)
)

# Test data preprocessing (no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Data Loading
train_data = train_datagen.flow_from_directory(
    directory=train_path,
    shuffle=True,
    classes=labels,
    target_size=(240, 240),
    subset="training",
    class_mode="categorical",
    batch_size=32,
)

val_data = train_datagen.flow_from_directory(
    directory=train_path,
    shuffle=True,
    classes=labels,
    target_size=(240, 240),
    subset="validation",
    class_mode="categorical",
    batch_size=32,
)

# --- Model Architecture Improvements ---

# Fine-tuning ResNet50 (Unfreeze some top layers)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(240, 240, 3))
for layer in base_model.layers[-30:]:  # Fine-tune the last 30 layers
    layer.trainable = True

# Enhanced Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)  # Increased units
x = BatchNormalization()(x)  # Added Batch Normalization
x = Dropout(0.5)(x)  # Added Dropout for regularization
predictions = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- Training Process Optimization ---

# EarlyStopping 콜백 (과적합 방지)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Learning Rate Scheduler (Reduce LR on plateau)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6
)

# Model Compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Initial higher LR
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()  # 모델 구조 출력

# Model Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,  # Increased epochs for fine-tuning
    callbacks=[early_stopping, lr_scheduler],  # Added LR scheduler
    verbose=1,
)

# --- Evaluation and Visualization ---

# 훈련 데이터에 대한 모델 평가
result = model.evaluate(train_data, verbose=1)
print("Training accuracy: ", result[1] * 100)

# 훈련 및 검증 정확도 시각화
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validate"], loc="upper left")
plt.grid()
plt.show()

# 테스트 데이터 로드 및 평가
test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    classes=labels,
    target_size=(240, 240),
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

loss, accuracy = model.evaluate(test_generator, verbose=1)
print("Test accuracy = %f; loss = %f" % (accuracy, loss))

# 모델 저장
model.save("model.keras")
