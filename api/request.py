import requests
import json 

url = 'https://model-app-score.onrender.com/'

response = requests.get(url=url)
if response:
    if response.status_code == 200: 
        try:
            print(response.text)
        except Exception as e:
            print(e)

# fazendo previsao 
data = {
    'score_3': 720.5,
    'risk_rate': 0.12,
    'credit_limit': 15000.0,
    'income': 5500.0,
    'n_defaulted_loans': 1.0,
    'n_issues': 2.0,
    'ok_since': 36.0,
    'n_bankruptcies': 0.0,
    'external_data_provider_credit_checks_last_year': 3.0,
    'external_data_provider_credit_checks_last_month': 1,
    'external_data_provider_credit_checks_last_2_year': 5.0,
    'reported_income': 5000,
    'score_rating_enc': 2,
    'risk_score': 0.15 
}

# funcao para inferencia 
def inference(data, url):
    post = requests.post(url, json=data)
    if post.status_code == 200:
        print(post.json())
    else:
        print(post.text)
        
# fazendo inferencia 
url_pred = 'https://model-app-score.onrender.com/predict'

if __name__ == "__main__":
    inference(data=data, url=url_pred)