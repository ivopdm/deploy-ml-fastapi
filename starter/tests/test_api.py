import json
import requests

def test_welcome_message():
    # response = requests.get("http://localhost:8000/")
    response = requests.get("https://salary-estimator.herokuapp.com")
    assert response.json() == {"message": "Welcome to my API"}
    assert response.status_code == 200

def test_predict_lte50k():
    data = {
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

    # response = requests.post("http://localhost:8000/predict", json=data)
    response = requests.post("https://salary-estimator.herokuapp.com/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"

def test_predict_gt50k():
    data = {
        "age": 32,
        "workclass": "Private",
        "fnlgt": 318647,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 48,
        "native_country": "United-States",
    }

    # response = requests.post("http://localhost:8000/predict", json=data)
    response = requests.post("https://salary-estimator.herokuapp.com/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"

def test_predict_with_invalid_data():
    data = {"age": "39", "workclass": "42", "fnlgt": "77516", 
        "education": "Bachelors", "education-num": "13", 
        "marital-status": "Never-married", "occupation": "Adm-clerical", 
        "relationship": "Husband", "race": "White","sex":"Male",
        "capital-gain": "2174", "capital-loss": "0", "hours-per-week": "40",
        "native-country": "United-States"}
    # response = requests.post("http://localhost:8000/predict", json=data)
    response = requests.post("https://salary-estimator.herokuapp.com/predict", json=data)
    assert response.status_code == 422