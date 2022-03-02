from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import pytest
import pandas as pd


@pytest.fixture(scope="session")
def data():
    cols = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'salary']
    data = [
        [
            39,
            "State-gov",
            77516,
            "Bachelors",
            13,
            "Never-married",
            "Adm-clerical",
            "Not-in-family",
            "White",
            "Male",
            2174,
            0,
            40,
            "United-States",
            " <=50K",
        ],
        [
            50,
            "Self-emp-not-inc",
            83311,
            "Bachelors",
            13,
            "Married-civ-spouse",
            "Exec-managerial",
            "Husband",
            "White",
            "Male",
            0,
            0,
            13,
            "United-States",
            " <=50K",
        ],
        [
            38,
            "Private",
            215646,
            "HS-grad",
            9,
            "Divorced",
            "Handlers-cleaners",
            "Not-in-family",
            "White",
            "Male",
            0,
            0,
            40,
            "United-States",
            " <=50K",
        ],
        [
            53,
            "Private",
            234721,
            "11th",
            7,
            "Married-civ-spouse",
            "Handlers-cleaners",
            "Husband",
            "Black",
            "Male",
            0,
            0,
            40,
            "United-States",
            " <=50K",
        ],
        [
            28,
            "Private",
            338409,
            "Bachelors",
            13,
            "Married-civ-spouse",
            "Prof-specialty",
            "Wife",
            "Black",
            "Female",
            0,
            0,
            40,
            "Cuba",
            " <=50K",
        ],
        [
            37,
            "Private",
            284582,
            "Masters",
            14,
            "Married-civ-spouse",
            "Exec-managerial",
            "Wife",
            "White",
            "Female",
            0,
            0,
            40,
            "United-States",
            " <=50K",
        ],
        [
            49,
            "Private",
            160187,
            "9th",
            5,
            "Married-spouse-absent",
            "Other-service",
            "Not-in-family",
            "Black",
            "Female",
            0,
            0,
            16,
            "Jamaica",
            " <=50K",
        ],
        [
            52,
            "Self-emp-not-inc",
            209642,
            "HS-grad",
            9,
            "Married-civ-spouse",
            "Exec-managerial",
            "Husband",
            "White",
            "Male",
            0,
            0,
            45,
            "United-States",
            " >50K",
        ],
        [
            31,
            "Private",
            45781,
            "Masters",
            14,
            "Never-married",
            "Prof-specialty",
            "Not-in-family",
            "White",
            "Female",
            14084,
            0,
            50,
            "United-States",
            " >50K",
        ],
        [
            42,
            "Private",
            159449,
            "Bachelors",
            13,
            "Married-civ-spouse",
            "Exec-managerial",
            "Husband",
            "White",
            "Male",
            5178,
            0,
            40,
            "United-States",
            " >50K",
        ],
    ]
    return pd.DataFrame(data, columns=cols)


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
        test,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
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
        test,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
    )

    # Train and save a model.
    clf = train_model(X_train, y_train)

    preds = inference(clf, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
