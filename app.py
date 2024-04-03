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
    color: grey;
}

.stApp h1, .stApp h2 {
    color: black;
}

.stButton > button {
    color: white;
    background-color: #0047AB;
}
</style>
""", unsafe_allow_html=True)

# Add a title and a quote
st.title("SMS Spam Detection Model")
st.markdown('<p style="color: black;">"SMS spam is a growing problem, but with the help of machine learning, we can fight back!"</p>', unsafe_allow_html=True)

# Add the SMS spam detection input and button
# Add an input box for the SMS
input_sms = st.markdown('<p style="color: black;">Enter the SMS...</p>', unsafe_allow_html=True)
input_sms = st.text_input("", help="Type your SMS here...", max_chars=500)

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

# Add the "About Us" section to the bottom of the web page in a single line
st.markdown("<p style='text-align: center; color: grey;'>Beware of Spam SMS ,They might empty your bank account</p>", unsafe_allow_html=True)
