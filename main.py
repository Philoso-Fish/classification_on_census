# Put the code for your API here.
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.model import inference
import numpy as np
import pandas as pd
import pickle

# instatiate the app
app = FastAPI()


class Body(BaseModel):
    age: List[int]
    workclass: List[str]
    fnlgt: List[int]
    education: List[str]
    education_num: List[int] = Field(alias='education-num')
    marital_status: List[str] = Field(alias='marital-status')
    occupation: List[str]
    relationship: List[str]
    race: List[str]
    sex: List[str]
    capital_gain: List[int] = Field(alias='capital-gain')
    capital_loss: List[int] = Field(alias='capital-loss')
    hours_per_week: List[int] = Field(alias='hours-per-week')
    native_country: List[str] = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": ["22"],
                "workclass": ["Private"],
                "fnlgt": ["77516"],
                "education": ["Masters"],
                'education-num': ["14"],
                'marital-status': ["Divorced"],
                "occupation": ["Exec-managerial"],
                "relationship": ["Husband"],
                "race": ["Black"],
                "sex": ["Female"],
                'capital-gain': ["2222"],
                'capital-loss': ["0"],
                'hours-per-week': ["35"],
                'native-country': ["Cuba"]
            }
        }


# Define GET to welcome users
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World"}


@app.post("/inference/")
async def predict(body: Body):
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
    df = pd.DataFrame(body.dict())
    # replace _ with - in columns
    df.columns = [col.replace("_", "-") for col in df.columns]

    X_categorical = df[cat_features].values
    X_continuous = df.drop(*[cat_features], axis=1)

    # load model and encoder
    file = open("classifier_fastapi.sav", 'rb')
    model = pickle.load(file)
    file = open("encoder_fastapi.sav", "rb")
    encoder = pickle.load(file)

    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    # return {"columns": X.shape}
    preds = inference(model, X)

    return {"predictions": preds.tolist()}
