import os
import glob
import pandas as pd
import numpy as np
import kagglehub
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

@st.cache_resource
def load_and_train():

    path = kagglehub.dataset_download("akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000")

    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
    if not csv_files:
        st.error("Could not find any CSV files in the downloaded dataset.")
        return None, None, None

    csv_path = max(csv_files, key=os.path.getsize)

    df = pd.read_csv(csv_path).dropna()

    df = df.sample(min(15000, len(df)))

    df.columns = [c.lower() for c in df.columns]

    if df['label'].dtype == 'object':
        df['label'] = df['label'].map({'ham': 0, 'spam': 1, 'phishing': 1, 'phish': 1})

    max_words, max_len = 5000, 150
    tokenizer = Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(df['text'].astype(str).values)

    X = pad_sequences(tokenizer.texts_to_sequences(df['text'].astype(str).values), maxlen=max_len)
    Y = df['label'].values

    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        SpatialDropout1D(0.3),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=2, batch_size=64, verbose=0)

    return model, tokenizer, max_len

# --- UI ---
st.set_page_config(page_title="Email Shield", page_icon="🛡️")
st.title("🛡️ LSTM Spam & Phish Detector")

with st.spinner("Loading dataset and training LSTM... this may take a minute."):
    model, tokenizer, max_len = load_and_train()

if model:
    user_input = st.text_area("Paste email text below:", height=200)
    if st.button("Run Security Check"):
        if user_input.strip():
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=max_len)
            prob = model.predict(padded)[0][0]

            st.divider()
            if prob > 0.5:
                st.error(f"🚨 ALERT: Potential Threat Detected (Confidence: {prob:.2%})")
            else:
                st.success(f"✅ CLEAN: This email appears safe (Confidence: {1-prob:.2%})")
        else:
            st.warning("Please enter some text to analyze.")
