# SMS Spam Detection System

This project is a web-based application that uses machine learning to classify SMS messages as spam or ham (not spam) in real time. Built using TensorFlow and Flask, the system provides a user-friendly interface for testing SMS messages and shows predictions with high accuracy.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [How to Use](#how-to-use)
- [Future Work](#future-work)
- [License](#license)

## Overview
This project is designed to identify unwanted SMS messages (spam) by analyzing message content using a pre-trained machine learning model. Users can enter a message through a web form, and the system instantly classifies it as spam or ham.

## Features
- **Real-Time Detection**: Quickly classifies messages as spam or ham.
- **High Accuracy**: Achieves approximately 98% accuracy on the dataset.
- **Easy-to-Use Interface**: Simple HTML form for message input.
- **Custom Spam Classifier**: Model is specifically trained on SMS data for spam detection.

## Technologies Used
- **Python**: Core programming language.
- **TensorFlow and Keras**: Used for model training and deployment.
- **Flask**: Web framework to serve the model as a web application.
- **Pandas and Numpy**: For data manipulation and preprocessing.
- **HTML/CSS**: Basic front-end setup for user interface.

## Architecture
1. **Data Preprocessing**: Processes and cleans SMS messages for the model.
2. **Model Training**: A neural network trained on a labeled SMS dataset (`spam.csv`) to classify messages.
3. **Model Deployment**: The trained model (`spam_model.h5`) is served through a Flask app.
4. **Prediction Pipeline**: Message input goes through preprocessing, model prediction, and result display.

## Getting Started

### Prerequisites
Make sure you have Python installed. Install dependencies by running:
```bash
pip install tensorflow flask pandas numpy
