import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ... rest of your code ...

# Add a background image
st.set_page_config(page_title="SMS Spam Detection Model", page_icon=":envelope:", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.istockphoto.com/photo/artificial-intelligence-processor-unit-powerful-quantum-ai-component-on-pcb-gm1464561797-497228177?utm_campaign=category_photos_bottom&utm_content=https%3A%2F%2Funsplash.com%2Fbackgrounds&utm_medium=affiliate&utm_source=unsplash&utm_term=backgrounds%3A%3A%3A");
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
    st.image("https://example.com/your-image.jpg", width=300)

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
