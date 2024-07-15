# Retinopathy Detection with ResNet50
This project aims to develop a model for detecting diabetic retinopathy from retinal images. Diabetic retinopathy is a serious eye condition that can lead to blindness if not detected early. The dataset used for this project is the APTOS 2019 Blindness Detection dataset from Kaggle.

## Table of Contents
1. Introduction
2. Dataset
3. Notebook Overview
4. Installation
5. Usage
6. Results
7. Conclusion
8. References
## Introduction
Diabetic retinopathy is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). Early detection through retinal image analysis can prevent severe vision loss.

This project uses a Convolutional Neural Network (CNN) with the ResNet50 architecture to classify retinal images into different stages of diabetic retinopathy.

## Dataset
The dataset used in this project is the APTOS 2019 Blindness Detection dataset from Kaggle. It contains images of retinas taken using fundus photography. The dataset includes the following classes:

- No DR (0)
- Mild (1)
- Moderate (2)
- Severe (3)
- Proliferative DR (4)
## Notebook Overview
The Jupyter Notebook included in this repository (Retinopathy detection.ipynb) covers the following steps:

- Importing Libraries: Loading necessary libraries for data processing, visualization, and model building.
- Data Import and Analysis: Loading the dataset, analyzing the distribution of classes, and visualizing sample images.
- Data Preprocessing: Steps to preprocess the images, including resizing, normalization, and augmentation.
- Model Construction: Building the ResNet50 model architecture using TensorFlow and Keras.
- Model Training: Training the model with appropriate hyperparameters, using techniques like learning rate scheduling and early stopping.
- Evaluation: Evaluating the model's performance using various metrics and visualizations like confusion matrices and ROC curves.
- User Interface: Demonstrating a simple GUI for real-time prediction (if applicable).
## Installation
To run the notebook, you need to have Python installed along with the necessary libraries:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
## Usage
Open the Jupyter Notebook and run the cells sequentially to reproduce the results. The notebook guides you through each step from data loading to model evaluation.

## Results
The model's performance is evaluated using several metrics, including accuracy, precision, recall, F1 score, and Cohen Kappa score. The results section in the notebook includes detailed visualizations and interpretations of these metrics.

## Conclusion
The ResNet50 model provides a reliable method for detecting diabetic retinopathy from retinal images. This project highlights the importance of deep learning in medical image analysis and demonstrates the effectiveness of CNNs in classifying complex image data.

## References
- APTOS 2019 Blindness Detection Dataset on Kaggle
- TensorFlow Documentation
- Keras Documentation
