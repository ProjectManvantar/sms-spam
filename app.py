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

# Add an About Us section on the left side of the web page
col1, col2 = st.beta_columns([1, 3])
with col1:
    st.subheader("About Us")
    st.write("We are a team of passionate data scientists and engineers dedicated to fighting SMS spam. Our model uses advanced machine learning techniques to accurately classify SMS messages as spam or not spam.")
    st.write('"We believe that technology can be used to make the world a better place, one SMS at a time."')

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

