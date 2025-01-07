from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle


#load Models
model=pickle.load(open('dtr.pkl','rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))


#Creating Flask app
app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Domain = request.form['Domain']
        Area = request.form['Area']
        Item = request.form['Item']
        Year = request.form['Year']
        Area_harvested = request.form['Area_harvested']
        Laying = request.form['Laying']
        Milk_Animals = request.form['Milk_Animals']
        Producing_AnimalsorSlaughtered = request.form['Producing_AnimalsorSlaughtered']
        Stocks = request.form['Stocks']
        Yield = request.form['Yield']
        YieldorCarcass_Weight = request.form['YieldorCarcass_Weight']

        feature = np.array([[Domain, Area, Item, Year, Area_harvested, Laying, Milk_Animals,
                             Producing_AnimalsorSlaughtered, Stocks, Yield, YieldorCarcass_Weight]])
        transform_feature = preprocessor.transform(feature)
        predicted_value = model.predict(transform_feature).reshape(1, -1)
        return render_template('index.html',predicted_value=predicted_value)

#Python Main
if __name__=='__main__':
    app.run(debug=True)
