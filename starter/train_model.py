# Script to train machine learning model.

import json
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append('starter/ml')

from model import train_model

from data import process_data
from model import compute_model_metrics, compute_model_metrics_by_slice

# Get root directory of the project.
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add code to load in the data using root_dir.
data = pd.read_csv(os.path.join(root_dir, 'data', 'census_cat.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", 
    training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
# Save the model to a pickle file
with open(os.path.join(root_dir, 'model', 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Save the encoder to a pickle file
with open(os.path.join(root_dir, 'model', 'encoder.pkl'), 'wb') as f:
    pickle.dump(encoder, f)

# Compute the model metrics using the test data.
precision, recall, fbeta = compute_model_metrics(y_test, model.predict(X_test))
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fbeta:.4f}")

# Compute the model metrics by slice using the test data
# and save the results in a json file.
metrics_by_slice = compute_model_metrics_by_slice(test, y_test, model.predict(X_test),'native-country')
with open(os.path.join(root_dir, 'screenshots', 'metrics_by_slice.json'), 'w') as f:
    json.dump(metrics_by_slice, f)