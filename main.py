from io import BytesIO
import pickle
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import pandas as pd
import sys
import requests

sys.path.append('starter/ml')

app = FastAPI()


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to my API"}


@app.post("/predict")
async def predict(data: Data):
    # root_dir = os.path.dirname(__file__)

    # model = pickle.load(
    #   open(os.path.join(root_dir,"model","model.pkl"), "rb"))
    # encoder = pickle.load(
    #   open(os.path.join(root_dir,"model","encoder.pkl"), "rb"))

    url_model_pickle = ("https://github.com/ivopdm/deploy-ml-fastapi/raw"
                        "/main/model/model.pkl")
    request_model_pickle = requests.get(url_model_pickle)
    model = pickle.load(BytesIO(request_model_pickle.content))

    url_encoder_pickle = ("https://github.com/ivopdm/deploy-ml-fastapi/raw"
                          "/main/model/encoder.pkl")
    request_encoder_pickle = requests.get(url_encoder_pickle)
    encoder = pickle.load(BytesIO(request_encoder_pickle.content))

    try:
        # data preprocessing
        df = pd.DataFrame(data.dict(), index=[0])

        # Define categorical features.
        cat_features = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ]

        # Proces the test data with the process_data function.
        X_categorical = df[cat_features].values
        X_continuous = df.drop(*[cat_features], axis=1)
        X_categorical = encoder.transform(X_categorical)

        # Remove unnamed columns.
        X_continuous = X_continuous.loc[:, ~X_continuous.columns.str.contains(
                                                                  "^Unnamed")]
        X = np.concatenate([X_continuous, X_categorical], axis=1)

        # do model inference
        pred = model.predict(X)
        res = "<=50K" if pred[0] == 0 else ">50K"

        # turn prediction into JSON
        return {"prediction": res}
    except BaseException:
        raise HTTPException(
            status_code=400,
            detail="Error during model inference")
