import pandas as pd
import numpy as np
import streamlit as st
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def load_model():

    data = pd.read_csv("email.csv")

    data = data.dropna()
    data['Category'] = data['Category'].str.strip().str.lower()
    data = data[data['Category'].isin(['ham', 'spam'])]
    data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

    corpus = []
    for msg in data['Message'].astype(str):
        msg = re.sub('[^a-zA-Z]', ' ', msg)
        msg = msg.lower()
        if msg.strip() == "":
            msg = "empty"
        corpus.append(msg)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    y = data['Category'].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, vectorizer

st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Email & SMS Detector")
st.write("Check whether a message is **Spam or Not Spam** using Machine Learning.")

with st.spinner("Training model..."):
    model, vectorizer = load_model()

user_input = st.text_area("Enter your message here:", height=200)

if st.button("Check Message"):
    if user_input.strip():

        msg = re.sub('[^a-zA-Z]', ' ', user_input)
        msg = msg.lower()

        vector = vectorizer.transform([msg])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0][1]

        st.divider()

        if prediction == 1:
            st.error(f"🚨 Spam Detected (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ Not Spam (Confidence: {1-prob:.2%})")

    else:
        st.warning("Please enter some text.")
