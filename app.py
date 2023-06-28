import streamlit as st
import pickle
import sklearn
import string
import json
import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie

# Preprocessing new email/sms

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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('EMAIL/SMS SPAM CLASSIFIER')

input_sms = st.text_input("Please enter the Email/SMS text below")

if st.button("Check"):
    transformed_sms = text_transformer(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Prediction
    result = model.predict(vector_input)[0]
    # Display
    if result == 1:
        st.subheader("SPAM ALERT!")
    else:
        st.subheader("NOT SPAM")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_coding = load_lottiefile('124955-notifications.json')
st_lottie(lottie_coding)
