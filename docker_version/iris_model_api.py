from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import mlflow.pyfunc

app=FastAPI(title="IRIS Model APIs")
mlflow.set_tracking_uri("http://34.93.117.209:8100")
model=mlflow.pyfunc.load_model("models:/iris_model_dt/latest")

class IrisInput(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length :  float
    petal_width : float

@app.get("/")
def read_root():
    return { "message": "Welcome to Iris API page for Assignment 6"}

@app.post("/predict")
def predict_species(data: IrisInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {
            "predicted class": prediction
            }

