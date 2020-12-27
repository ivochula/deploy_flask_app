import numpy as np
from flask import Flask, request, render_template
import pickle
import os

# app name
app = Flask(__name__)


# load the save model
def load_model():
    return pickle.load(open('dataloggers_model.pkl', 'rb'))


# home page
@app.route('/')
def home():
    return render_template('index.html')


# predict the result and return it
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]

    values = [np.array(features)]

    model = load_model()
    result = round(float(model.predict(values)), 2)

    return render_template('index.html', output='Rental price is {} BRL'.format(result))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
