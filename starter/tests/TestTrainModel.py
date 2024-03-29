import sys

import pandas as pd
sys.path.append('../ml')
from model import compute_model_metrics, train_model, compute_model_metrics_by_slice

import numpy as np

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