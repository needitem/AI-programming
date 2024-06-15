import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 레이블 정의
labels = ["good", "bad"]

# 훈련 및 테스트 데이터 경로
train_path = "Data/train"
test_path = "Data/test"

# 데이터 증강 및 전처리 (훈련용)
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.1,  # 10%를 검증 데이터로 사용
)

# 테스트 데이터 전처리 (증강 없음)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# 훈련 데이터 로드
train_data = datagen.flow_from_directory(
    directory=train_path,
    shuffle=True,
    classes=labels,
    target_size=(227, 227),  # AlexNet에 맞게 이미지 크기 조정 (227x227)
    subset="training",  # 훈련 데이터 지정
    class_mode="categorical",
    batch_size=32,
)

# 검증 데이터 로드
val_data = datagen.flow_from_directory(
    directory=train_path,
    shuffle=True,
    classes=labels,
    target_size=(227, 227),  # AlexNet에 맞게 이미지 크기 조정 (227x227)
    subset="validation",  # 검증 데이터 지정
    class_mode="categorical",
    batch_size=32,
)

# AlexNet 모델 구축
model = Sequential()
model.add(
    Conv2D(96, (11, 11), strides=(4, 4), activation="relu", input_shape=(227, 227, 3))
)
model.add(MaxPooling2D((3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))
model.add(Conv2D(384, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(384, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))  # 2개 클래스: good, bad

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 학습률 조정
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()  # 모델 구조 출력

# EarlyStopping 콜백 (과적합 방지)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# 모델 훈련
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stopping],
    verbose=1,
)

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
test_generator = test_datgen.flow_from_directory(
    directory=test_path,
    classes=labels,
    target_size=(227, 227),  # AlexNet에 맞게 이미지 크기 조정 (227x227)
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

loss, accuracy = model.evaluate(test_generator, verbose=1)
print("Test accuracy = %f; loss = %f" % (accuracy, loss))

# 모델 저장
model.save("alexnet_model.keras")
