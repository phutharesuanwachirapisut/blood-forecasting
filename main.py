#!/usr/bin/env python3
# command for open python:source venv/bin/activate
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# โหลดโมเดล (จากไฟล์ .pkl ที่คุณ save ไว้)
model = joblib.load("linear_model.pkl")

# สร้าง FastAPI app
app = FastAPI()

# Define input schema

class InputData(BaseModel):
    ICU: float
    ER: float
    Oncology: float
    Surgery: float
    Cardiac: float
    Transplant: float
    Trauma: float
    All_Patients: float

    year: int
    month: int

    RBCs_Issued_lag1: float
    RBCs_Issued_lag7: float
    RBCs_Issued_lag14: float

    RBCs_Issued_rollmean7: float
    RBCs_Issued_rollstd7: float
    RBCs_Issued_rollmean14: float
    RBCs_Issued_rollstd14: float

    RBCs_Issued_ewma_0_1: float  
    RBCs_Issued_ewma_0_3: float  
    RBCs_Issued_ewma_0_5: float  
    
    forecast_naive: float
    forecast_seasonal_naive: float
    
feature_cols = [
    "ICU", "ER", "Oncology", "Surgery", "Cardiac", "Transplant", "Trauma", "All_Patients",
    "year", "month",
    "RBCs_Issued_lag1", "RBCs_Issued_lag7", "RBCs_Issued_lag14",
    "RBCs_Issued_rollmean7", "RBCs_Issued_rollstd7",
    "RBCs_Issued_rollmean14", "RBCs_Issued_rollstd14",
    "RBCs_Issued_ewma_0_1", "RBCs_Issued_ewma_0_3", "RBCs_Issued_ewma_0_5",
    "forecast_naive", "forecast_seasonal_naive"
]

@app.get("/")
def home():
    return {"message": "Model API is running!"}

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    # เรียง features ตามลำดับที่ train
    X = np.array([[input_dict[col] for col in feature_cols]])
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}