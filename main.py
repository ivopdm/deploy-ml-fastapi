import os
import pickle
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
import pandas as pd
import sys

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
    root_dir = os.path.dirname(__file__)
    # load pickle model
    model = pickle.load(os.path.join(root_dir,"model","model.pkl"))
    # encoder = pickle.load(open(os.path.join(root_dir,"model","encoder.pkl"), "rb"))
    # lb = LabelBinarizer()
    encoder = pickle.load(os.path.join(root_dir,"model","encoder.pkl"))
    
    try:
        # data preprocessing
        # df = pd.DataFrame(data.dict(),index=[0])

        # Replace - by _ in dictionary values.
        data_dict = data.dict()
        data_dict = {k.replace("-", "_"): v for k, v in data_dict.items()}
        df = pd.DataFrame(data_dict, index=[0])

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
        X_categorical = X[cat_features].values
        X_continuous = X.drop(*[cat_features], axis=1)
        # Remove unnamed columns.
        X_continuous = X_continuous.loc[:, ~X_continuous.columns.str.contains("^Unnamed")]
        X = np.concatenate([X_continuous, X_categorical], axis=1)

        # do model inference
        result = model.predict(X)
        return {"prediction": result.tolist()}
        # return {"prediction": ">50k"}
    except:
        raise HTTPException(status_code=400, detail="Error during model inference")