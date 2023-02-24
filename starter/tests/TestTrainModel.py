from model import compute_model_metrics, train_model
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append('starter/ml')


class TestTrainModel(unittest.TestCase):
    def test_train_model(self):
        # Generate some fake training data and labels
        X_train = np.random.rand(10, 5)
        y_train = np.random.randint(2, size=10)

        # Train the model
        model = train_model(X_train, y_train)

        # Check that the model has been successfully trained
        self.assertIsNotNone(model)
        self.assertIsInstance(model, LogisticRegression)

    def test_compute_model_metrics(self):
        # Generate some fake known labels and predicted labels
        y = np.array([0, 0, 1, 1])
        preds = np.array([0, 1, 0, 1])

        # Compute the model metrics
        precision, recall, fbeta = compute_model_metrics(y, preds)

        # Check that the correct values are returned
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 0.5)
        self.assertEqual(fbeta, 0.5)

    def test_compute_model_metrics_zero_division(self):
        # Generate some fake known labels and predicted labels
        y = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 0, 0])

        # Compute the model metrics
        precision, recall, fbeta = compute_model_metrics(y, preds)

        # Check that the correct values are returned
        self.assertEqual(precision, 0)
        self.assertEqual(recall, 0)
        self.assertEqual(fbeta, 0)


if __name__ == '__main__':
    unittest.main()
