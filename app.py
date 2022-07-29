# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 21:56:39 2022

@author: 91931
"""
import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model = pickle.load(open('linear_regression_mvi_project_2.pkl','rb')) 

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))
    
    prediction = model.predict([[exp]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted Price for given Requirements of SqFt Area is : {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug = True)
