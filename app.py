import text_preprocess
from text_preprocess import text_preprocess
import streamlit as st
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# Load the model
model = load_model('artifact/model.h5')

# Load the tokenizer
# Make sure to save your tokenizer after training, or recreate it with the same parameters
with open('artifact/Tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)



# Function to preprocess input text
def preprocess_text(text):
    text_preprocess(text)
    # Integer encode the text
    encoded_text = tokenizer.texts_to_sequences([text])
    # Pad the sequence
    padded_text = pad_sequences(encoded_text, maxlen=100, padding='post')
    return padded_text

# Streamlit app
st.title("Text Classification with GRU Model")
st.write("Enter some text to classify:")

# Text input
user_input = st.text_area("Input Text")

if st.button("Classify"):
    if user_input:
        # Preprocess the input text
        padded_input = preprocess_text(user_input)
        
        # Make prediction
        prediction = model.predict(padded_input)
        predicted_class = (prediction[0][0] > 0.5).astype(int)  # Assuming binary classification
        
        # Display the result
        if predicted_class == 1:
            st.success("The text is classified as Positive.")
        else:
            st.success("The text is classified as Negative.")
    else:
        st.warning("Please enter some text to classify.")