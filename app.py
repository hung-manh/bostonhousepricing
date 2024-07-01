import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar  =pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get the data from the POST request.
    data = request.json['data']

    # # Make prediction using the model
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    prediction = regmodel.predict(new_data)

    # Take the first value of prediction    
    output = prediction[0]
    return jsonify(output)

@app.route('/predict', methods=['POST'])
def predict12():
    data = [float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1, -1)) 
    prediction = regmodel.predict(final_input)[0]
    # Can return the prediction in any format
    return render_template('home.html', prediction_text2hello='The House preice prediction is: {}'.format(prediction),
                            prediction_text2='The House preice prediction is: {}'.format(prediction))


if __name__ == '__main__':
    app.run(port=5000, debug=True)