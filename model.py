import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import pickle

df = pd.read_csv('https://raw.githubusercontent.com/ivochula/deploy_flask_app/main/locacao_datalogger.csv')

X = df.drop(columns=['Total', 'Preco_dia'])
y = df['Preco_dia']

rf = ExtraTreesRegressor(max_depth=16, min_samples_leaf=2, n_estimators=1000)
rf.fit(X, y)

filename = 'dataloggers_model.pkl'
pickle.dump(rf, open(filename, 'wb'))