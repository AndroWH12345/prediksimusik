"""
Flask server
"""
import pickle
import os
import random
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template
from playsound import playsound

import Metadata
from Metadata import getmetadata

app = Flask(__name__)
    
SAVED_MODEL = pickle.load(open("./model_dtm.pkl","rb"))
@app.route("/predict", methods=["POST"])
def predict():


    # get audio file
    audio_file = request.files["UploadedAudio"]

    # random string of digits for file name
    file_name = str(random.randint(0, 100000))

    # save the file locally
    audio_file.save(file_name)

    # invoke the genre recognition service
    
    grs = getmetadata(file_name)

    #Process
    df = pd.read_csv("./music_genre_dataset.csv")
    df['label'] = df['label'].astype('category')

    lookup_genre_name = dict(zip(df.label.unique(), df.label.unique()))   

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    X = df.iloc[:,1:27]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
    #scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)

    from sklearn.svm import SVC
    clf = SVC(kernel = 'linear', C=10).fit(X_train_scaled, y_train)
    clf.score(X_test_scaled, y_test)


    # make prediction
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    
    d1 = np.array(grs)  
    data1 = scaler.transform([d1])
    prediction = SAVED_MODEL.predict(data1)
 

    # remove the .wav file
    os.remove(file_name)

    # message to be displayed on the html webpage
    prediction_message = f"""
    The song is predicted to be in the {prediction} genre.
    """
    return render_template("index.html", prediction_text=prediction_message)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=False)
