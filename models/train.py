import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import joblib

df = pd.read_csv('C:\\Users\\souza\\OneDrive\\Área de Trabalho\\Risk Nubank\\data\\data_tratado.csv')
# dropando colunas inviaveis para treino
column_drop = ['Unnamed: 0', 'score_rating_enc']
df = df.drop(column_drop, axis=1)
x = df.drop('target_default', axis=1)
y = df['target_default']

# split 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# seleçao de atributos 
cat_atribs = ['score_rating', 'situation']
num_atribs = ['score_3', 'risk_rate', 'credit_limit', 'income', 'n_defaulted_loans', ' n_issues', 'ok_since',
            'n_bankruptcies']

# criando pipeline e column transformer 

column_tranf = ColumnTransformer([('one_hot',OneHotEncoder(handle_unknown='ignore'), cat_atribs)
                                ])
pipe = Pipeline([
    ('transformer', column_tranf),
    ('model', XGBClassifier(
        gamma=3,  
        max_depth=8,
        max_delta_step=1,
        subsample=0.6803547999898337,
        reg_lambda=0.7557940213385035
    ))
])
model = pipe.fit(x_train, y_train)
y_pred = model.predict(x_test)

# metricas 
print(f'recall: {recall_score(y_test, y_pred)}')
print(f'acc: {accuracy_score(y_test, y_pred)}')
print(f'precision: {precision_score(y_test, y_pred)}')
print(f'f1_score: {f1_score(y_test, y_pred)}')

# salvando o modelo 
joblib.dump(pipe, 'XGBoost.pkl')
print(x.columns)
print(x.info())
