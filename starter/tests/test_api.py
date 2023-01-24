import requests

def test_welcome_message():
    # response = requests.get("http://localhost:8000/")
    response = requests.get("https://salary-estimator.herokuapp.com:5000/")
    assert response.json() == {"message": "Welcome to my API"}
    assert response.status_code == 200

def test_predict():
    data = {"age": 28, "workclass": "Private", "fnlgt": 30912, "education": "HS-grad", 
    "education_num": 9, "marital_status": "Married-civ-spouse", "occupation": "Other-service",
    "relationship": "Husband", "race": "White", "sex":"Male", "capital_gain": 0, 
    "capital_loss": 0, "hours_per_week": 43, "native_country": "United-States"}
    # response = requests.post("http://localhost:8000/predict", json=data)
    response = requests.post("https://salary-estimator.herokuapp.com:5000/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_with_invalid_data():
    data = {"age": "39", "workclass": "42", "fnlgt": "77516", 
        "education": "Bachelors", "education-num": "13", 
        "marital-status": "Never-married", "occupation": "Adm-clerical", 
        "relationship": "Husband", "race": "White","sex":"Male",
        "capital-gain": "2174", "capital-loss": "0", "hours-per-week": "40",
        "native-country": "United-States"}
    # response = requests.post("http://localhost:8000/predict", json=data)
    response = requests.post("https://salary-estimator.herokuapp.com:5000/predict", json=data)
    assert response.status_code == 422