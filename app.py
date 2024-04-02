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
</style>
""", unsafe_allow_html=True)

# Add a title and a subheader
st.title("SMS Spam Detection Model")
st.subheader("Made with ❤️‍🔥 by Shrudex👨🏻‍💻")

# Add an input box for the SMS
input_sms = st.text_input("Enter the SMS", placeholder="Type here...", max_chars=500)

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
