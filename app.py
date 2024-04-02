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
st.markdown("""
<style>
.stApp {
background-image: url("https://wallpapershome.com/images/pages/pic_h/16745.jpg");
background-attachment: fixed;
background-size: cover;
}
.stButton > button {
    color: white;
    background-color: #0047AB;
}
</style>
""", unsafe_allow_html=True)

# Add a title and a quote
st.title("SMS Spam Detection Model")
st.subheader('"SMS spam is a growing problem, but with the help of machine learning, we can fight back!"')

# Partition the web page into two columns with equal width
col1, col2 = st.columns([1, 2])

# Add the "About Us" section to the left column
with col1:
    st.subheader("About Us")
    st.write("We are a team of passionate individuals dedicated to fighting SMS spam. Our goal is to provide an easy-to-use and effective solution for detecting and filtering out unwanted SMS messages.")
    st.write("Our model is trained on a large dataset of SMS messages and uses advanced machine learning techniques to accurately classify messages as spam or not spam.")

# Add the SMS spam detection input and button to the right column
with col2:
    # Add an input box for the SMS
    input_sms = st.text_input("Enter the SMS", help="Type your SMS here...", max_chars=500)

    # Add a button to predict
    if st.button('Predict', key='predict'):
        # Check if the input is not empty
        if input_sms:
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
        else:
            st.warning("Please enter an SMS.")
