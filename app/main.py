from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd 
import joblib
import os 
import sys
sys.path.append(r'C:\\Users\\souza\\OneDrive\\Área de Trabalho\\Risk Nubank')

# setando aplicação
app = FastAPI()
model_path = os.path.join('DecisionTree.pkl')
# carregando modelo 
model = joblib.load(model_path)
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
        risk_score: float
@app.get('/')
def home():
        return{'api no ar'}
@app.post('/predict')
def predict(data:ModelInput):
        try:
                df = pd.DataFrame([data.dict()])
                y_pred = model.predict(df)
                y_proba = model.predict_proba(df)[:,1]
                
                return {'saida:': int(y_pred[0])}
        except Exception as e:
                return {f'erro {str(e)}'}
        
