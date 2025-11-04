## Rupali Pragati

## Overview

This project focuses on detecting spam emails and messages using machine learning classification techniques.
It demonstrates how text data can be processed, vectorized, and classified into spam or non-spam categories using the Naive Bayes algorithm.

## Objectives

Preprocess and clean text-based email datasets.

Convert messages into numerical form using TF-IDF vectorization.

Train and evaluate a spam detection model using Naive Bayes.

Test the model on custom user input messages.

Dataset

Dataset Source: SMS Spam Collection Dataset on Kaggle

The dataset is automatically downloaded in the script using KaggleHub.

Project Workflow

Data Collection: Import dataset automatically from Kaggle using KaggleHub.

Data Cleaning: Remove missing values, rename columns, and map labels.

Text Processing: Transform text into numerical features using TF-IDF.

Model Building: Train a Multinomial Naive Bayes classifier.

Evaluation: Measure performance using accuracy, classification report, and confusion matrix.

Prediction: Allow user to test custom messages for spam detection.

## Results

The model achieved high accuracy in distinguishing between spam and non-spam messages.

TF-IDF vectorization combined with Naive Bayes performed efficiently on text classification tasks.

Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

KaggleHub

Run the Project
# Install dependencies
pip install -r requirements.txt

# Run the project
python email_spam_detector.py