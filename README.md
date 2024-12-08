# Network Traffic Classification with Random Forest

This project uses a **Random Forest** classifier to classify network traffic as either "normal" or "malicious" based on various features, such as **time**, **protocol**, **source**, and **destination**. The model is trained using labeled network traffic data and can be used for network security analysis.

## Project Overview

- **Objective**: Classify network traffic as either normal or malicious.
- **Approach**: 
    - Preprocess data by encoding categorical variables (Protocol, Source, Destination).
    - Train a Random Forest classifier on labeled network traffic data.
    - Evaluate the model’s performance using accuracy, classification report, and confusion matrix.
- **Output**: A trained Random Forest model that predicts whether network traffic is normal or malicious.

## Installation

To run this project, you will need **Python 3.6+** and the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

### Install required dependencies
Use the following command to install project & dependencies.

```bash
git clone https://github.com/IlluminateDreamer/network-traffic-classifier-model
cd network-traffic-classifier-model
pip install -r requirements.txt --break-system-packages
```

### Usage
Follow the instructions to use the `network-traffic-class`.

```
python3 process_dataset.py
python3 train_model.py
python3 test.py
```
## Files

Here are the key files in this project:

- **`training_data.csv`**: Raw dataset containing network traffic information for training.
- **`traffic_classifier_model.pkl`**: Trained Random Forest model, saved after training.
- **`le_protocol.pkl, le_source.pkl, le_destination.pkl`**: Label encoders for transforming categorical data into numerical format.
- **`labeled_traffic_data.csv`**: Dataset with the added label column (`bad_packet`) after preprocessing.
- **`process_dataset.py`**: Script for cleaning and preparing the dataset (label encoding and feature extraction).
- **`train_model.py`**: Script for training the Random Forest classifier.
- **`test_model.py`**: Script for testing the trained model on new/unlabeled data.

## Future Improvements

- **Real-time Network Monitoring**: Integrate the model into a real-time network monitoring system for anomaly detection.
- **Experiment with Other Models**: Try other machine learning models (e.g., SVM, XGBoost) to improve classification accuracy.
- **Feature Engineering**: Explore additional features (e.g., packet size, source IP address patterns) for enhancing the model.


