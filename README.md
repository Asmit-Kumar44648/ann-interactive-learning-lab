# ANN Interactive Learning Lab

## Overview
ANN Interactive Learning Lab is a web-based Artificial Neural Network
demonstration built using TensorFlow/Keras and Streamlit.

The application allows users to modify neural network parameters,
train a classification model, examine learning curves, evaluate
predictions, and experiment with new inputs.

## Objectives
- Understand the structure of an artificial neural network.
- Experiment with hidden layers and neurons.
- Observe the effect of learning rate, epochs, and batch size.
- Visualize training and validation accuracy and loss.
- Evaluate classification performance using test metrics.
- Make predictions through an interactive web interface.

## Technology Stack
- Python
- TensorFlow/Keras
- Streamlit
- NumPy and Pandas
- Matplotlib
- Scikit-learn

## Dataset
The application uses the Iris flower dataset available through
Scikit-learn. It contains 150 samples, four numerical features,
and three flower classes.

## Model
- Input: 4 numerical features
- Hidden layers: Configurable dense layers
- Hidden activation: ReLU
- Output: 3 neurons with Softmax
- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Test split: 20%
- Random seed: 42

## Run the Application
Install the dependencies:

    pip install -r requirements.txt

Start Streamlit:

    streamlit run app.py

## Deployment
The application can be deployed using Streamlit Community Cloud
from this GitHub repository.

## Evaluation
The application displays test accuracy, test loss, training and
validation curves, a confusion matrix, a classification report,
and interactive predictions.

## Author
Student: [Your Name]
Course: Artificial Neural Networks
Activity: Activity 2
