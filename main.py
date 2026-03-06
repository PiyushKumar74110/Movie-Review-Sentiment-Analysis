# ---------------- IMPORT LIBRARIES ---------------- #

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="wide"
)


# ---------------- GLASSMORPHISM CSS ---------------- #

st.markdown(
    """
    <style>

    .stApp{
        background: linear-gradient(135deg,#141E30,#243B55);
        color:white;
    }

    /* Glass card */
    .glass{
        background: rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 25px;
        backdrop-filter: blur(10px);
        border:1px solid rgba(255,255,255,0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    textarea{
        background: rgba(255,255,255,0.1) !important;
        color:white !important;
        border-radius:10px !important;
    }

    .stButton>button{
        background: linear-gradient(90deg,#00c6ff,#0072ff);
        border:none;
        color:white;
        border-radius:8px;
        padding:10px 20px;
        font-weight:600;
    }

    .stProgress > div > div{
        background-color:#00c6ff;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- LOAD MODEL ---------------- #

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model("simple_rnn_imdb.h5")


# ---------------- FUNCTIONS ---------------- #

def preprocessing_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


def predict_sentiment(review):

    processed_input = preprocessing_text(review)
    prediction = model.predict(processed_input)

    score = float(prediction[0][0])

    sentiment = "Positive" if score > 0.5 else "Negative"

    return sentiment,score


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Project Information")

st.sidebar.write("Model : Simple RNN")
st.sidebar.write("Dataset : IMDB Movie Reviews")
st.sidebar.write("Framework : TensorFlow / Keras")
st.sidebar.write("Interface : Streamlit")

st.sidebar.markdown("---")

st.sidebar.write(
"""
This application uses Natural Language Processing
to classify movie reviews as positive or negative.
"""
)


# ---------------- MAIN TITLE ---------------- #

st.title("IMDB Movie Review Sentiment Analysis")

st.write(
"""
A sentiment classification system powered by a Recurrent Neural Network.
Enter a movie review below and the model will analyze its sentiment.
"""
)


# ---------------- GLASS CARD ---------------- #

st.markdown('<div class="glass">',unsafe_allow_html=True)

st.subheader("Enter Movie Review")

user_input = st.text_area(
    "Type or paste a review",
    height=180
)

st.subheader("Example Reviews")

col1,col2 = st.columns(2)

with col1:
    if st.button("Positive Example"):
        user_input = "The movie had amazing acting and a fantastic storyline."

with col2:
    if st.button("Negative Example"):
        user_input = "The film was boring and completely predictable."


st.subheader("Sentiment Prediction")

if st.button("Analyze Review"):

    if user_input.strip()!="":

        sentiment,confidence = predict_sentiment(user_input)

        st.markdown("### Prediction Result")

        if sentiment=="Positive":
            st.success("Predicted Sentiment : Positive")

        else:
            st.error("Predicted Sentiment : Negative")

        st.markdown("### Model Confidence")

        st.progress(confidence)

        st.write(f"Confidence Score : {confidence:.3f}")

    else:

        st.warning("Please enter a movie review.")


st.markdown('</div>',unsafe_allow_html=True)


# ---------------- FOOTER ---------------- #

st.markdown("---")

st.write(
"Demonstration of Sentiment Analysis using Recurrent Neural Networks."
)
