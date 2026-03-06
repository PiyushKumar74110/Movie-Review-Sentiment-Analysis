#import libaries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the word index
word_index=imdb.get_word_index()
reverse_word_index={value : key for key ,value in word_index.items()}

#load the pre-trained model with ReLU activation
model=load_model('simple_rnn_imdb.h5')

#helper function and function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#process user-input
def preprocessing_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2) + 3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#pediction function
def predict_sentiment(review):
    preprocessed_input=preprocessing_text(review)

    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]



#streamlit
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')


#user input
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocessing_text(user_input)

    #make prediction
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    #Display the result
    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

