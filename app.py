import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

# train model ...

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake News Detection Model")

news = st.text_area("Enter News Article")

if st.button("Predict"):
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == "FAKE":
        st.error("FAKE NEWS")
    else:
        st.success("REAL NEWS")
