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
    result2 = round((features[0] * features[1]) * result, 2)
    day = int(features[1])
    logger = int(features[0])

    return render_template('index.html', days='For {0} days and {1} datalogger(s)'.format(day, logger),
                           output='Rental price per logger is {} BRL for one day'.format(result),
                           output2='Total price for the rental is {} BRL'.format(result2))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
