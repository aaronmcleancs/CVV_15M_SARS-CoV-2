import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Define image size and number of classes
IMAGE_SIZE = 224
NUM_CLASSES = 3

# Define class labels corresponding to model predictions
class_labels = ['COVID', 'Normal', 'Viral Pneumonia']

# Load the model
model = keras.models.load_model('chest_xray_classifier_final.h5')

# Recompile the model (optional but useful for evaluation)
model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Preprocessing function for single image
def preprocess_image(image_path):
    img_array = cv2.imread(image_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    resized_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(resized_array, axis=0), resized_array  # Return processed and original image

# Predict function to generate predictions and visualize results
def predict_and_visualize(image_path):
    processed_image, original_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get class with the highest probability
    confidence = prediction[0][predicted_class]  # Confidence for predicted class

    # Human-readable class label
    predicted_label = class_labels[predicted_class]

    # Generate output image with prediction and model info
    generate_output_image(image_path, original_image, predicted_label, confidence, prediction)

def generate_output_image(image_path, original_image, predicted_label, confidence, prediction):
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)  # Show the original image
    plt.axis('off')

    # Add predicted class, confidence, and model info as text overlay
    plt.title(f"Prediction: {predicted_label} ({confidence * 100:.2f}% confidence)", fontsize=14, color='black')
    plt.figtext(0.5, 0.01, f"Model: chest_xray_classifier_final.h5", ha="center", fontsize=12)
    
    # Save the result as a new image
    output_image_path = f"{os.path.splitext(image_path)[0]}_prediction.png"
    plt.savefig(output_image_path)
    plt.show()

    print(f"Generated output image saved at: {output_image_path}")

# Define the folder with the unlabeled images
unlabeled_images_dir = '/Users/aaronmclean/Desktop/Work/GitHub/SARSCOV19_NN/model_test'

# Iterate through images and predict
for img_file in os.listdir(unlabeled_images_dir):
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        image_path = os.path.join(unlabeled_images_dir, img_file)
        print(f"Processing {img_file}...")
        predict_and_visualize(image_path)
