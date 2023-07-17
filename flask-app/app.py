
from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

classifier = pickle.load(open("flask-app/model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('/index.html', user_data = [0,0,0])

@app.route('/predict', methods= ["POST","GET"])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = classifier.predict(features)
    
    output = "{0}".format(prediction[0])
    if int(output) != 1 : 
        return render_template("/index.html",
         prediction_text = "Your not capable to buy a bike.\n Prediction of {}.".format(output),user_data = int_features  )
    else :
        return render_template("/index.html", prediction_text = "Your capable to buy a bike.\n Prediction of  {}.".format(output),user_data = int_features)
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
 