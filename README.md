COVID-19 X-ray Classification

<table>
  <tr>
    <td align="center">
      <strong>COVID-19</strong><br>
      <img src="https://github.com/user-attachments/assets/a954b5d5-bd59-4add-95e5-00d88c1f753e" alt="COVID-19 Prediction" width="300"/>
    </td>
    <td align="center">
      <strong>Normal</strong><br>
      <img src="https://github.com/user-attachments/assets/c5e0d25f-aa2b-4796-98ef-c7748c1c2bf5" alt="Control" width="300"/>
    </td>
    <td align="center">
      <strong>Viral Pneumonia</strong><br>
      <img src="https://github.com/user-attachments/assets/677cbb73-330c-41d9-8444-6e6f65fdf160" alt="Viral Pneumonia Prediction" width="300"/>
    </td>
  </tr>
</table>

![Training History CVV](https://github.com/user-attachments/assets/592afbd7-3a29-4c3c-bb47-b4811fe076e3)


Implementation of a Convolutional Neural Network (CNN) for classifying chest X-ray images into three categories: COVID-19, Normal, and Viral Pneumonia using TensorFlow, AWS, and Keras for training and evaluation on apple M-series CPUs

Download Dataset: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Download Weights: https://www.mediafire.com/file/q6n3zrrj1cqxkml/chest_xray_classifier_final.h5/file

Usage: 
pip install -r requirements.txt
python SARS-CoV-2_Tensor.py
python SARS-CoV-2_RunModel.py path/to/image.jpg
