#import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model("simple_rnn_imdb.h5")

# Decode review
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3, "?") for i in encoded_review])

# Preprocess input text
def preprocessing_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    processed_input = preprocessing_text(review)
    prediction = model.predict(processed_input)
    score = float(prediction[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score


# ---------------- UI CONFIGURATION ---------------- #

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="centered"
)

# Header
st.title("IMDB Movie Review Sentiment Analysis")
st.markdown(
"""
This application uses a **Recurrent Neural Network (RNN)** trained on the **IMDB movie review dataset**
to classify whether a review expresses a **positive** or **negative** sentiment.
"""
)

# Sidebar Information
st.sidebar.header("Project Information")

st.sidebar.markdown(
"""
**Model:** Simple RNN  
**Dataset:** IMDB Movie Reviews  
**Framework:** TensorFlow / Keras  
**Interface:** Streamlit  

The model processes a movie review, converts it into a sequence of word indices,
and predicts sentiment using a trained neural network.
"""
)

# Input Section
st.subheader("Enter Movie Review")

user_input = st.text_area(
    "Type or paste a movie review below",
    height=180
)

# Example Reviews
st.subheader("Example Reviews")

col1, col2 = st.columns(2)

with col1:
    if st.button("Load Positive Example"):
        st.session_state.review = "The movie had excellent performances and a very engaging story."

with col2:
    if st.button("Load Negative Example"):
        st.session_state.review = "The film was slow, predictable and not enjoyable."

# Prediction Button
st.subheader("Sentiment Prediction")

if st.button("Analyze Review"):

    if user_input.strip() != "":
        sentiment, confidence = predict_sentiment(user_input)

        st.markdown("### Prediction Result")

        if sentiment == "Positive":
            st.success("Predicted Sentiment: Positive")
        else:
            st.error("Predicted Sentiment: Negative")

        st.markdown("### Model Confidence")
        st.progress(confidence)

        st.write(f"Confidence Score: {confidence:.3f}")

    else:
        st.warning("Please enter a movie review before running the analysis.")

# Footer
st.markdown("---")
st.markdown(
"""
Developed as a demonstration of **Natural Language Processing using Recurrent Neural Networks**.
"""
)
