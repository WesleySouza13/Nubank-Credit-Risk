from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd 
import joblib

app = FastAPI()
model = joblib.load('C:\Users\souza\OneDrive\Área de Trabalho\Risk Nubank\DecisionTree.pkl')
class ModelInput(BaseModel):
        score_3:float
        risk_rate:float
        credit_limit:float
        income:float
        n_defaulted_loans:float
        n_issues:float
        ok_since:float
        n_bankruptcies:float
        external_data_provider_credit_checks_last_year:int
        external_data_provider_credit_checks_last_month:int
        external_data_provider_credit_checks_last_2_year:int
        reported_income:int
        score_rating_enc:int
@app.get('/')
def home():
        return{'api no ar'}

@app.post('/predict')
def predict(data:ModelInput):
        if model is None:
                print('modelo nao encontrado')
        try:
                df = pd.DataFrame([data.dict()])
                y_pred = model.predict(df)
                y_proba = model.predict_proba(df)[:,1]


                return {f'saida: {y_pred} e probabilidade: {y_proba[0]*100:.2f}%'}
        except Exception as e:
                return {e}