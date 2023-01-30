import os
import pickle
from fastapi import FastAPI, HTTPException
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
    capital_gain: int 
    capital_loss: int 
    hours_per_week: int 
    native_country: str 

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
    lb = LabelBinarizer()
    
    try:
        # data preprocessing
        df = pd.DataFrame(data.dict(),index=[0])
        # df = pd.json_normalize(data)
        # print(df.head())
        # print(data.dict())

        # Proces the test data with the process_data function.
        X, _, _, _ = process_data(
            df, categorical_features=cat_features, label=None, 
            training=False, encoder=encoder, lb=lb
        )

        # do model inference
        result = model.predict(X)
        return {"prediction": result.tolist()}
        # return {"prediction": ">=50k"}
    except:
        raise HTTPException(status_code=400, detail="Error during model inference")