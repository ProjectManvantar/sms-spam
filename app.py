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
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpaperaccess.com/full/2819254.jpg");
        background-attachment: fixed;
        background-size: cover
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.title("SMS Spam Detection Model")
    st.write("*Great to See YOU Here!!!*")
    st.write("This app predicts whether an SMS is spam or not based on the text of the message.")

with col2:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://wallpaperaccess.com/full/2819254.jpg");
            background-position: center;
            background-repeat: no-repeat;
            background-size: contain;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

input_sms = st.text_input("Enter the SMS", placeholder="Type your SMS here...", key="input")

if st.button('Predict', key="predict"):
    with st.spinner('Processing...'):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        st.write("**Processed Text:**", transformed_sms)

        # 2. vectorize
        vector_input = tk.transform([transformed_sms])

        # 3. predict
        result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
        st.write("The SMS you entered is predicted to be spam.")
    else:
        st.header("Not Spam")
        st.write("The SMS you entered is predicted to be not spam.")

    # Add a feedback mechanism
    user_feedback = st.selectbox("Was the prediction correct?", ["Yes", "No"])
    if user_feedback:
        st.write(f"Thanks for your feedback! You said the prediction was {user_feedback}.")
