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
background-image: url("https://wallpapertag.com/wallpaper/full/4/e/a/142810-free-download-4k-nature-wallpapers-3840x2160-computer.jpg");
background-attachment: fixed;
background-size: cover;
}
.stButton > button {
    color: white;
    background-color: #0047AB;
}
</style>
""", unsafe_allow_html=True)

# Add a title, a subheader, and an image of Shankar Babu
st.title("SMS Spam Detection Model")
st.subheader("A project by Shrudex, under the guidance of Shankar Babu 🌟")
st.image("https://wallpapertag.com/wallpaper/full/4/e/a/142810-free-download-4k-nature-wallpapers-3840x2160-computer.jpg", width=300)

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

# Add some information about Shankar Babu
st.subheader("About Shankar Babu 👨‍🏫")
st.write("Shankar Babu is our guide for this project. He is a highly experienced and knowledgeable professor in the field of computer science. His guidance and support have been invaluable to us throughout this project.")
