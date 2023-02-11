from io import BytesIO
import json
import os
import pickle
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
import pandas as pd
import sys
import requests

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
sys.path.append('starter/ml')

from data import process_data
# Get root directory of the project.
# root_dir = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

class Data(BaseModel):
    age:int
    workclass: str
    fnlgt: int
    education: str 
    education_num:int 
    marital_status: str
    occupation: str
    relationship: str 
    race: str
    sex: str
    capital_gain: float 
    capital_loss: float 
    hours_per_week: float 
    native_country: str 



@app.get("/")
def read_root():
    return {"message": "Welcome to my API"}

@app.post("/predict")
async def predict(data: Data):
    url_model_pickle = "https://github.com/ivopdm/deploy-ml-fastapi/raw/main/model/model.pkl"
    request_model_pickle = requests.get(url_model_pickle)
    model = pickle.load(BytesIO(request_model_pickle.content))

    url_encoder_pickle = "https://github.com/ivopdm/deploy-ml-fastapi/raw/main/model/encoder.pkl"
    request_encoder_pickle = requests.get(url_encoder_pickle)
    encoder = pickle.load(BytesIO(request_encoder_pickle.content))

    try:
        # data preprocessing
        df = pd.DataFrame(data.dict(),index=[0])        

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
        X_continuous = X_continuous.loc[:, ~X_continuous.columns.str.contains("^Unnamed")]
        X = np.concatenate([X_continuous, X_categorical], axis=1)

        # do model inference
        pred = model.predict(X)   
        res = "<=50K" if pred[0] == 0 else ">50K"

        # turn prediction into JSON
        return {"prediction": res}
    except:
        raise HTTPException(status_code=400, detail="Error during model inference")