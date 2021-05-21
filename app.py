# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:00:14 2021

@author: KanaseCom
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    if(output==1):
        return render_template('index.html', prediction_text='Loan is Approved')
    else:
        return render_template('index.html', prediction_text='Loan is not Approved')

if __name__ == "__main__":
    app.run(debug=True)