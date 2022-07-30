# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:45:35 2022

@author: 91931
"""

import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model = pickle.load(open('linearregression_houseprices_project2.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp1 = float(request.args.get('exp1'))
    exp2 = float(request.args.get('exp2'))
    exp3 = float(request.args.get('exp3'))
    exp4 = float(request.args.get('exp4'))
    exp5 = float(request.args.get('exp5'))
    exp6 = float(request.args.get('exp6'))
    
    prediction = model.predict([[exp1,exp2,exp3,exp4,exp5,exp6]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted Price for given Requirements of SqFt Area is : {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug = True)
