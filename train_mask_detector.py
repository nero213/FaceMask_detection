import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set paths
TRAIN_DIR = "Dataset/train"
IMG_SIZE = 100

# Load and preprocess data
def load_data(data_dir):
    categories = ["with_mask", "without_mask"]
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,    img))
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            data.append([resized, class_num])
    np.random.shuffle(data)
    X = np.array([item[0] for item in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X / 255.0
    y = tf.keras.utils.to_categorical([item[1] for item in data], num_classes=2)
    return X, y

# Build model
def build_model():
    model = Sequential([
        Conv2D(200, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((3, 3)),
        Conv2D(100, (3, 3), activation="relu"),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Train
X, y = load_data(TRAIN_DIR)
model = build_model()
model.fit(X, y, batch_size=60, epochs=60, validation_split=0.2)
model.save("mask_detector.h5")
print("Model trained and saved!")