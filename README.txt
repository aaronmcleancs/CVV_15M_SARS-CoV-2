COVID-19 X-ray Classification

This project implements a Convolutional Neural Network (CNN) for classifying chest X-ray images into three categories: COVID-19, Normal, and Viral Pneumonia. The code utilizes TensorFlow 2.x, Keras, and other libraries for efficient training and evaluation.

Features:

GPU Acceleration: Leverages GPU for faster training if available.
Mixed Precision: Utilizes mixed precision training (float16) to improve training speed and memory usage on compatible hardware.
Data Preprocessing: Loads and preprocesses chest X-ray images for the model.
Model Training: Trains a CNN model with image augmentation (optional, not implemented in the provided code).
Evaluation: Evaluates the model's performance on a hold-out test set and generates various reports.
TensorBoard Integration: Logs training metrics for visualization in TensorBoard.
Early Stopping: Implements early stopping to prevent overfitting.
Model Saving: Saves the trained model for future use.
Training History Visualization: Plots training and validation accuracy/loss curves.
Classification Reports: Generates confusion matrix and classification report for detailed performance analysis.
Results Saving: Saves test accuracy, confusion matrix, and classification report to a text file.
Requirements:

Python 3.x
TensorFlow 2.x
Keras
OpenCV (cv2)
NumPy (np)
Pandas (pd)
Scikit-learn
Matplotlib
tqdm (optional, for progress bar)
zipfile
Instructions:

Download Dataset:

Replace the placeholder path /usr/colab/COVID-19_Radiography_Dataset.zip with the actual path to your downloaded COVID-19 X-ray dataset compressed in a ZIP archive.
Run the Script:

Open a terminal or command prompt and navigate to the directory containing this script (README.md and the Python code).
Execute the script using Python: python main.py
Outputs:

The script will generate the following files in the same directory:
chest_xray_classifier_final.h5: The trained model file.
training_history.png: A plot visualizing training and validation accuracy/loss curves.
classification_results.txt: A text file containing test accuracy, confusion matrix, and classification report.
