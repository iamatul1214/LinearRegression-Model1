import numpy as np

from Operations import model_Operations
from flask import Flask, render_template
from flask import request, redirect , url_for
from linearRegression import LinearRegressionModel as lr
import jinja2
import pickle


jinja_env=jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))

mo=model_Operations()

app=Flask(__name__)
model=pickle.load(open('Linearregression_example.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/pandasprofileEDA', methods=['GET','POST'])
def eda():
    return render_template('testlr.html')

@app.route('/predict',methods=['POST'])
def predict():
    features=[float(f) for f in request.form.values()]

    features_ready=[np.array(features)]

    prediction=model.predict(features_ready)

    predicted_value=round(prediction[0][0],1)


    return render_template('index.html',predicted_text='The predicted value of Air temperature is {}'.format(predicted_value))


if __name__=="__main__":
    app.run(debug=True)