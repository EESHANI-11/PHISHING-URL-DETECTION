#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import joblib

warnings.filterwarnings('ignore')
from feature import FeatureExtraction

gbc = joblib.load("pickle/model.pkl")


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,31) 
        print(f"Features shape: {x.shape}")  # Debugging output
        if x.shape[1] != 31:
            return f"Error: Model expects 31 features but received {x.shape[1]}"

        print(f"Total features extracted: {x.shape[1]}")
        print("Features received:", x)

         # Ensure x has the correct number of features
        if x.shape[1] > gbc.n_features_in_:
           print(f"⚠️ Extra feature detected! Removing last column to match {gbc.n_features_in_} expected features.")
           x = x[:, :gbc.n_features_in_]  # Keep only the first 30 columns


        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)