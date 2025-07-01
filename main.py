from fastapi import FastAPI
from spam_data import SMS
import pandas as pd 
import numpy as np
import pickle
import sklearn
import string
import json
import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = FastAPI(
    title = "Spam Detector",
    description = "This model classifies if a SMS is spam or ham"
)

@app.get("/")
def index():
    return "Welcome to Spam-Detector"

def text_transformer(text):
    # Converting to lower case
    text = text.lower()

    # Performing Tokenization
    text = nltk.word_tokenize(text)

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Removing stopwords and punctuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Performing stemming
    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

ps = PorterStemmer()

tfidf = pickle.load(open('/Users/ahmad/Documents/GitHub/Spam-Detector/vectorizer.pkl','rb'))
model = pickle.load(open('/Users/ahmad/Documents/GitHub/Spam-Detector/model.pkl','rb'))

@app.post("/predict")
def make_predictions(sms: SMS):
    # Preprocess
    transformed_sms = text_transformer(sms.text)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Return response
    return {"prediction": "SPAM" if result == 1 else "NOT SPAM"}
