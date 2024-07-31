from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    df1 = pd.read_csv("Application/dataset/hate_speech_dataset.csv")

    df_with_keys = pd.concat([df1], keys=["hate_speech"])
    dataset = df_with_keys
    
    dataset = dataset[["CONTENT", "CLASS"]]

    dataset_X = dataset["CONTENT"]
    dataset_Y = dataset["CLASS"]

    corpus = dataset_X
    cv = TfidfVectorizer()
    X = cv.fit_transform(corpus).toarray()

    model = open("Application/model/model.pkl", "rb")
    clf = pickle.load(model)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('result.html', prediction = my_prediction)
    
if __name__ == '__main__':
    app.run(debug = True)