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
    df1 = pd.read_csv("ML_Web_App/dataset/Youtube01-Psy.csv")
    df2 = pd.read_csv("ML_Web_App/dataset/Youtube02-KatyPerry.csv")
    df3 = pd.read_csv("ML_Web_App/dataset/Youtube03-LMFAO.csv")
    df4 = pd.read_csv("ML_Web_App/dataset/Youtube04-Eminem.csv")
    df5 = pd.read_csv("ML_Web_App/dataset/Youtube05-Shakira.csv")

    df_with_keys = pd.concat([df1, df2, df3, df4, df5], keys=["Psy", "KatyPerry", "LMFAO", "Eminem", "Shakira"])
    dataset = df_with_keys
    
    dataset = dataset[["CONTENT", "CLASS"]]

    dataset_X = dataset["CONTENT"]
    dataset_Y = dataset["CLASS"]

    corpus = dataset_X
    cv = TfidfVectorizer()
    X = cv.fit_transform(corpus).toarray()

    model = open("ML_Web_App/model/model.pkl", "rb")
    clf = pickle.load(model)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('result.html', prediction = my_prediction)
    
if __name__ == '__main__':
    app.run(debug = True)