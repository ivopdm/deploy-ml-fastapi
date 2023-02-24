from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # logistic regression model hiperparameters tuning
    model = LogisticRegression()
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring="f1")
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

# Write a function that outputs the performance of the model on slices
# of the data


def compute_model_metrics_by_slice(X, y, preds, slice_column):
    """
    Computes the model metrics by slice.

    Inputs
    ------
    X : pd.Dataframe
        Data used for prediction.
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    slice_column : str
        Column used for slicing.
    Returns
    -------
    metrics_by_slice : dict
        Dictionary of metrics by slice.
    """
    metrics_by_slice = {}
    for slice_value in X[slice_column].unique():
        slice_mask = X[slice_column] == slice_value
        slice_preds = preds[slice_mask]
        slice_y = y[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        metrics_by_slice[slice_value] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }
    return metrics_by_slice
