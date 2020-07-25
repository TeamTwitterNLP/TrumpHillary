from flask import Flask, render_template, redirect, request
from sklearn.externals import joblib

app = Flask(__name__) 
scaler = joblib.load("scaler.save")

@app.route("/",methods=['GET', 'POST'])
def index():

    if request.method == "GET":
        return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST': #this block is only entered when the form is submitted
        my_input = request.form.to_dict()
        print(my_input)
        

        return "Hello there"









from nltk.corpus import stopwords


if __name__ == "__main__":
    app.run(debug=True)