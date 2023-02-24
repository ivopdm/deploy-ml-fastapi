import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

# Get root directory of the project.
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    salary: str
    example = {"age": 39, "workclass": "State-gov", "fnlgt": 77516,
               "education": "Bachelors", "education-num": 13,
               "marital-status": "Never-married", "occupation": "Adm-clerical",
               "relationship": "Husband", "race": "White", "sex": "Male",
               "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
               "native-country": "United-States", "salary": "<=50K"}


@app.get("/")
def read_root():
    return {"message": "Welcome to my API"}


@app.post("/predict")
def predict(data: Data):
    # load pickle model
    model = pickle.load(
        open(
            os.path.join(
                root_dir,
                "model",
                "model.pkl"),
            "rb"))
    try:
        # data preprocessing
        data = pd.DataFrame(data.dict())
        # do model inference
        result = model.predict(data)
        return {"prediction": result.tolist()}
    except BaseException:
        raise HTTPException(
            status_code=400,
            detail="Error during model inference")
