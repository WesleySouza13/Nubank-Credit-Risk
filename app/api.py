from pydantic import BaseModel
from fastapi import FastApi
import pandas as pd 
import joblib

app = FastApi()
model = joblib.load('C:\\Users\\souza\\OneDrive\\Área de Trabalho\\Risk Nubank\\XGBoost.pkl')
class ModelInput(BaseModel):
        score_3:float
        risk_rate:float
        credit_limit:float
        income:float
        n_defaulted_loans:float
        n_issues:float
        ok_since:float
        n_bankruptcies:float
        score_rating:str
        situation:str
@app.get('/')
def home():
    return{'api no ar'}