import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Add a background image
st.set_page_config(page_title="SMS Spam Detection Model", page_icon=":envelope:", layout="wide", initial_sidebar_state="expanded")

# Add a sidebar
st.sidebar.title("About")
st.sidebar.write("Made with ❤️‍🔥 by Shrudex👨🏻‍💻")

# Main page
st.title("SMS Spam Detection Model")
st.write("Enter your SMS below to check if it's spam or not.")

input_sms = st.text_area("Enter the SMS", height=150)

if st.button('Predict', key='predict'):
    with st.spinner("Analyzing SMS..."):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tk.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")
