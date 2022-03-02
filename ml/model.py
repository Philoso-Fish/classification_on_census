from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np


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
    clf = DecisionTreeClassifier(random_state=22)
    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

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
    return model.predict(X)

def test_on_slices(model, data, col, label, encoder, lb, categorical_features):
    """
    Tests on slices of categorical data

    :param model: sklearn, trained model
    :param data: DataFrame, preprocessed data
    :param col: str, Column name of categorical data
    :param label: str, Label column
    :param encoder: OneHotEncoder
    :param lb: Label binarizer
    :param categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    :return: list(tuple(value:str, precision:float, recall:float, fbeta:float))
    """
    result = []

    # get unique values
    ar_unique = data[col].unique()

    for val in ar_unique:
        df_temp = data.loc[data[col] == val]

        # transform data
        y = df_temp[label]
        X = df_temp.drop([label], axis=1)

        X_categorical = X[categorical_features].values
        X_continuous = X.drop(*[categorical_features], axis=1)
        X_categorical = encoder.transform(X_categorical)
        y = lb.transform(y.values).ravel()
        X = np.concatenate([X_continuous, X_categorical], axis=1)

        # get predictions and metrics
        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)

        result.append((val, precision, recall, fbeta))

    return result
