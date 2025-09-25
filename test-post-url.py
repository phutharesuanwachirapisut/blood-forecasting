import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "ICU": 10,
    "ER": 20,
    "Oncology": 15,
    "Surgery": 12,
    "Cardiac": 8,
    "Transplant": 5,
    "Trauma": 7,
    "All_Patients": 77,
    "year": 2025,
    "month": 9,
    "RBCs_Issued_lag1": 50,
    "RBCs_Issued_lag7": 55,
    "RBCs_Issued_lag14": 60,
    "RBCs_Issued_rollmean7": 52,
    "RBCs_Issued_rollstd7": 3,
    "RBCs_Issued_rollmean14": 54,
    "RBCs_Issued_rollstd14": 4,
    "RBCs_Issued_ewma_0_1": 53,
    "RBCs_Issued_ewma_0_3": 52,
    "RBCs_Issued_ewma_0_5": 51,
    "forecast_naive": 50,
    "forecast_seasonal_naive": 58
}

res = requests.post(url, json=data)
print(res.json())