# importações 
import sys
sys.path.append(r'C:\\Users\\souza\\OneDrive\\Área de Trabalho\\Risk Nubank')
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from FeatureEng import FeatureEng
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_curve
import joblib

csv = 'C:\\Users\\souza\\OneDrive\\Área de Trabalho\\Risk Nubank\\data\\data_newTarget.csv'

# importando dados e fazendo split
df = pd.read_csv(csv)
x = df.drop('y_target', axis=1)
y = df['y_target']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y)

# pipeline 
pipe = Pipeline([
    ('featureeng', FeatureEng()),
    ('scaler', MaxAbsScaler()),
    ('model', DecisionTreeClassifier(max_depth=40,
                                    min_samples_split=38,
                                    min_samples_leaf=20,
                                    max_features='log2',
                                    criterion='gini',
                                    random_state=42, 
                                    class_weight='balanced'))
])

# definição do modelo 
model = pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
y_prob = pipe.predict_proba(x_test)[:,1]
# metricas 
print(f'logloss: {log_loss(y_test, y_pred)}')
print(f'brier score: {brier_score_loss(y_test, y_prob)}')

# calculando ks
fpr, tpr, _ = roc_curve(y_test, y_prob)
ks = max(tpr-fpr)
print(f'ks: {ks}')

# salvando o modelo 
joblib.dump(pipe, 'DecisionTree.pkl')
