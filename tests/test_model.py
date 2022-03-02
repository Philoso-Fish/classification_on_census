from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import pytest
import pandas as pd


@pytest.fixture(scope='session')
def data():
    return pd.read_csv("processed_census.csv")


def test_train(data):
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

    # Train and save a model.
    clf = train_model(X_train, y_train)

    assert clf


def test_inference(data):
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
    X_test, y_test, encoder_test, lb_test = process_data(
        test, categorical_features=cat_features, encoder=encoder, lb=lb, label="salary", training=False
    )

    # Train and save a model.
    clf = train_model(X_train, y_train)

    preds = inference(clf, X_test)
    assert preds.shape == y_test.shape


def test_metrics(data):
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
    X_test, y_test, encoder_test, lb_test = process_data(
        test, categorical_features=cat_features, encoder=encoder, lb=lb, label="salary", training=False
    )

    # Train and save a model.
    clf = train_model(X_train, y_train)

    preds = inference(clf, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert precision
    assert recall
    assert fbeta
