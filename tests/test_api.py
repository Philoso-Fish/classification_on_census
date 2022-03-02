from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World"}


def test_negative_pred():
    body = {
        "age": ["22"],
        "workclass": ["Private"],
        "fnlgt": ["77516"],
        "education": ["Masters"],
        "education-num": ["14"],
        "marital-status": ["Divorced"],
        "occupation": ["Exec-managerial"],
        "relationship": ["Husband"],
        "race": ["Black"],
        "sex": ["Female"],
        "capital-gain": ["2222"],
        "capital-loss": ["0"],
        "hours-per-week": ["35"],
        "native-country": ["Cuba"],
    }
    r = client.post("/inference/", json=body)
    assert r.status_code == 200
    assert r.json() == {"predictions": [0]}


def test_positive_pred():
    body = {
        "age": ["45"],
        "workclass": ["Self-emp-not-inc"],
        "fnlgt": ["209642"],
        "education": ["HS-grad"],
        "education-num": ["9"],
        "marital-status": ["Married-civ-spouse"],
        "occupation": ["Exec-managerial"],
        "relationship": ["Husband"],
        "race": ["White"],
        "sex": ["Male"],
        "capital-gain": ["0"],
        "capital-loss": ["0"],
        "hours-per-week": ["45"],
        "native-country": ["United-States"],
    }
    r = client.post("/inference/", json=body)
    assert r.status_code == 200
    assert r.json() == {"predictions": [1]}
