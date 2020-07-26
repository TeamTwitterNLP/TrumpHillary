from flask import Flask, render_template, redirect, request, url_for, session
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import os


value = None
img1= None
text= None

app = Flask(__name__) 
app.secret_key = "supersecretkey"
app.config['TESTING'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
scaler = joblib.load("scaler.save")

@app.route("/", methods=['GET', 'POST'])
def index():

    global value
    global img1
    

    
    
    return render_template("index.html", value=value, img1=img1, text=text)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST': #this block is only entered when the form is submitted
        my_input = request.form.to_dict()
        my_input_array = pd.Series(my_input["input text"])
        my_input_array2= my_input["input text"]

        print(my_input_array)

        prediction = scaler.predict(my_input_array)

        print("My Prediction: ", prediction)

        global value
        global img1
        global text
        

        if prediction == 0:
            value = "Trump"
            img1 = "https://media3.oakpark.com/Images/2/2/42153/2/1/2_2_42153_2_1_350x700.jpg"
            text= my_input_array2
        elif prediction == 1:
            value = "Hillary"
            img1 = "https://i.guim.co.uk/img/media/030b519118d984e4c1fd4d7b4c6fff60e6228852/886_27_1926_1155/master/1926.jpg?width=605&quality=45&auto=format&fit=max&dpr=2&s=21ee8e9f1e2c79102d94263ec16f961c"
            text= my_input_array2
        
        return redirect(url_for("index"))


@app.before_request
def make_session_permanent():
    session.permanent = False






from nltk.corpus import stopwords


if __name__ == "__main__":
    app.run(debug=True)