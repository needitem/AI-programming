import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
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
    target_size=(240, 240),
    subset="training",  # 훈련 데이터 지정
    class_mode="categorical",
    batch_size=32,
)

# 검증 데이터 로드
val_data = datagen.flow_from_directory(
    directory=train_path,
    shuffle=True,
    classes=labels,
    target_size=(240, 240),
    subset="validation",  # 검증 데이터 지정
    class_mode="categorical",
    batch_size=32,
)

# ResNet50 모델 구축
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(240, 240, 3))

# 기본 모델 레이어 동결 (가중치 업데이트 방지)
for layer in base_model.layers:
    layer.trainable = False

# 분류 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)  # 2개 클래스: good, bad

model = Model(inputs=base_model.input, outputs=predictions)

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
    target_size=(240, 240),
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

loss, accuracy = model.evaluate(test_generator, verbose=1)
print("Test accuracy = %f; loss = %f" % (accuracy, loss))

# 모델 저장
model.save("model.keras")
