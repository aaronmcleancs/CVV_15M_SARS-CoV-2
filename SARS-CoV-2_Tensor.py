import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import zipfile

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available. Enabling memory growth.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found. Using CPU.")

tf.keras.mixed_precision.set_global_policy('mixed_float16')

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
NUM_CLASSES = 3

print(f"Configuration:\n"
      f"IMAGE_SIZE: {IMAGE_SIZE}\n"
      f"BATCH_SIZE: {BATCH_SIZE}\n"
      f"EPOCHS: {EPOCHS}\n"
      f"NUM_CLASSES: {NUM_CLASSES}\n")

def unzip_data(zip_path, extract_to):
    print(f"Unzipping data from {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzipping complete.")

def load_and_preprocess_data(data_dir):
    print("Loading and preprocessing data...")
    data = []
    target = []
    categories = {'COVID': 'covid', 'Normal': 'normal', 'Viral Pneumonia': 'pneumonia'}

    for category in categories:
        path = os.path.join(data_dir, category, 'images')
        class_num = categories[category]
        print(f"Processing {category} images...")
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                resized_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
                data.append(resized_array)
                target.append(class_num)
            except Exception as e:
                print(f"Error processing image {img}: {e}")

    print("Data loading complete.")
    return np.array(data), np.array(target)

def create_model():
    print("Creating model...")
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

class DetailedTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir='logs', **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = self.model.optimizer.lr.numpy()
        super().on_epoch_end(epoch, logs)

def main():
    print("Starting the COVID-19 X-ray Classification process...")

    
    zip_path = '/usr/colab/COVID-19_Radiography_Dataset.zip'
    extract_to = '/usr/colab/COVID-19_Radiography_Dataset'

    unzip_data(zip_path, extract_to)

    X, y = load_and_preprocess_data(extract_to)
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = keras.utils.to_categorical(y)
    print(f"Encoded labels shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    model = create_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled. Summary:")
    model.summary()

    tensorboard_callback = DetailedTensorBoard(log_dir="./logs")
    early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback, early_stopping],
        verbose=1
    )

    print("Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc:.4f}')

    model.save('chest_xray_classifier_final.h5')
    print("Model saved as 'chest_xray_classifier_final.h5'")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred_classes)
    cr = classification_report(y_true, y_pred_classes, target_names=le.classes_)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    with open('classification_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(cr)

    print("Results saved to 'classification_results.txt'")

if __name__ == "__main__":
    main()
