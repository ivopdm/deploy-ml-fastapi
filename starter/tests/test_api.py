import requests
import sys

import pandas as pd
sys.path.append('starter/ml')
from model import compute_model_metrics, train_model, compute_model_metrics_by_slice

import numpy as np

def test_welcome_message():
    # response = requests.get("http://localhost:8000/")
    response = requests.get("https://salary-estimator.herokuapp.com")
    assert response.json() == {"message": "Welcome to my API"}
    assert response.status_code == 200


def test_predict_lte50k():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    # response = requests.post("http://localhost:8000/predict", json=data)
    response = requests.post(
        "https://salary-estimator.herokuapp.com/predict",
        json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_predict_gt50k():
    data = {
        "age": 32,
        "workclass": "Private",
        "fnlgt": 318647,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 48,
        "native_country": "United-States",
    }

    # response = requests.post("http://localhost:8000/predict", json=data)
    response = requests.post(
        "https://salary-estimator.herokuapp.com/predict",
        json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_predict_with_invalid_data():
    data = {
        "age": "39",
        "workclass": "42",
        "fnlgt": "77516",
        "education": "Bachelors",
        "education-num": "13",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": "2174",
        "capital-loss": "0",
        "hours-per-week": "40",
        "native-country": "United-States"}
    # response = requests.post("http://localhost:8000/predict", json=data)
    response = requests.post(
        "https://salary-estimator.herokuapp.com/predict",
        json=data)
    assert response.status_code == 422

# test prediction for null values


def test_predict_with_null_values():
    data = {
        "age": None,
        "workclass": None,
        "fnlgt": None,
        "education": None,
        "education_num": None,
        "marital_status": None,
        "occupation": None,
        "relationship": None,
        "race": None,
        "sex": None,
        "capital_gain": None,
        "capital_loss": None,
        "hours_per_week": None,
        "native-country": None}
    # response = requests.post("http://localhost:8000/predict", json=data)
    response = requests.post(
        "https://salary-estimator.herokuapp.com/predict",
        json=data)
    assert response.status_code == 422

# test prediction for missing variables


def test_predict_with_missing_variables():
    data = {
        "education": None,
        "education_num": None,
        "marital_status": None,
        "occupation": None,
        "relationship": None,
        "race": None,
        "sex": None,
        "capital_gain": None,
        "capital_loss": None,
        "hours_per_week": None,
        "native-country": None}

    # response = requests.post("http://localhost:8000/predict", json=data)

    response = requests.post(
        "https://salary-estimator.herokuapp.com/predict",
        json=data)
    assert response.status_code == 422

def test_train_model():
    # Generate some fake training data and labels
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(2, size=10)

    # Train the model
    model = train_model(X_train, y_train)

    # Check that the model has been successfully trained
    assert model != None

def test_compute_model_metrics():
    # Generate some fake known labels and predicted labels
    y = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 0, 1])

    # Compute the model metrics
    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Check that the correct values are returned
    assert precision == 0.5
    assert recall == 0.5
    assert fbeta == 0.5

# test compute_model_metrics_by_slice function
def test_compute_model_metrics_by_slice():
    # Generate some fake known labels and predicted labels
    d = {'col1': ['A', 'A', 'B', 'B'], 'col2': [0, 0, 0, 0], 'col3': [0, 0, 0, 0], 'col4': [1, 1, 1, 1]}
    X = pd.DataFrame(data=d)
    y = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 0, 1])
    slice_column = 'col1'

    # Compute the model metrics
    metrics_by_slice = compute_model_metrics_by_slice(X, y, preds, slice_column)

    # Check that the correct values are returned
    assert metrics_by_slice['A']['precision'] == 1.0
    assert metrics_by_slice['A']['recall'] == 1.0
    assert metrics_by_slice['A']['fbeta'] == 1.0
    assert metrics_by_slice['B']['precision'] == 1.0
    assert metrics_by_slice['B']['recall'] == 0.5
    assert metrics_by_slice['B']['fbeta'] == 0.6666666666666666
