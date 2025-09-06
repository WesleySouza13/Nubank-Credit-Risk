from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd 
import joblib
import os 

# setando aplicação
app = FastAPI()
# carregando modelo 
model = joblib.load('DecisionTree.pkl')
class ModelInput(BaseModel):
        score_3:float
        risk_rate:float
        credit_limit:float
        income:float
        n_defaulted_loans:float
        n_issues:float
        ok_since:float
        n_bankruptcies:float
        external_data_provider_credit_checks_last_year:float
        external_data_provider_credit_checks_last_month:int
        external_data_provider_credit_checks_last_2_year:float
        reported_income: int
        score_rating_enc:int
        risk_score: float
@app.get('/')
def home():
        return{'api no ar [modelo de application score] - criada por wesley matos'}
@app.post('/predict')
def predict(data:ModelInput):
        try:
                input_data = data.model_dump()
                df = pd.DataFrame([input_data])
                y_pred = model.predict(df)                
                return {'previsao [0 para adimplente e 1 para inadimplente]': int(y_pred[0])}
        except Exception as e:
                return {"erro": str(e)}
