import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
import pandas as pd
import io
from utils.scraper import extract_text_from_url
from utils.whatsapp_parser import extract_messages_from_whatsapp
st.set_page_config(page_title="Fake News & WhatsApp Detector", layout="centered")
import streamlit as st

# Logo and Header
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-top: -30px;
        color: #1a73e8;
    }
     <style>
    .footer {
        margin-top: 60px;
        text-align: center;
        color: grey;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.image("https://img.icons8.com/color/96/fake-news.png", width=80)
st.markdown("<div class='main-title'>Fake News & WhatsApp News Detector</div>", unsafe_allow_html=True)

# Load models
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News & WhatsApp Detector", layout="wide")
st.title("📰 Fake News & WhatsApp News Detector")
# Dark mode toggle
# WhatsApp-style dark theme toggle
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"], key="theme_selector")

if theme == "Dark":
    st.markdown(
        """
        <style>
        /* App background */
        .stApp {
            background-color: #121212;
            color: #e0e0e0;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1e1e1e;
        }

        /* Input widgets */
        .css-1cpxqw2, .css-1y4p8pa, .css-1fv8s86, .css-1t6c9ts {
            background-color: #2a2a2a !important;
            color: #ffffff !important;
            border-radius: 6px;
        }

        /* Markdown and text */
        .css-qrbaxs, .css-10trblm {
            color: #f1f1f1 !important;
        }

        /* Headers */
        h1, h2, h3, h4 {
            color: #ffffff;
        }

        /* Buttons */
        .stButton>button {
            background-color: #2e7d32;
            color: white;
            border-radius: 6px;
        }

        /* Selectbox, text areas, etc. */
        .stSelectbox, .stTextInput, .stTextArea {
            background-color: #2a2a2a !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    # Optional: reset style for light mode
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            color: black;
        }
        section[data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to predict
@st.cache_data
def predict_news(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()
    label = "Fake" if prediction == 1 else "Real"
    return label, confidence

# Plot chart
def plot_prediction_distribution(predictions):
    labels = ['Real', 'Fake']
    values = [predictions.count(0), predictions.count(1)]
    colors = ['green', 'red']

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')

    st.subheader("🧠 Prediction Distribution")
    st.pyplot(fig)

# Option selection
option = st.sidebar.selectbox("Choose an input method:", ["Type or Paste Text", "Analyze from URL", "Upload WhatsApp Chat (.txt)"])

if option == "Type or Paste Text":
    user_input = st.text_area("Paste news article or message here:", height=200)
    if st.button("Analyze") and user_input:
        label, confidence = predict_news(user_input)
        st.success(f"Prediction: {label} ({confidence * 100:.2f}%)")

elif option == "Analyze from URL":
    url = st.text_input("Paste article URL:")
    if st.button("Extract & Analyze") and url:
        try:
            content = extract_text_from_url(url)
            if content:
                st.text_area("Extracted Text:", content[:1000] + ("..." if len(content) > 1000 else ""))
                label, confidence = predict_news(content)
                st.success(f"Prediction: {label} ({confidence * 100:.2f}%)")
            else:
                st.warning("Could not extract content from the URL.")
        except Exception as e:
            st.error(f"Error: {e}")

elif option == "Upload WhatsApp Chat (.txt)":
    uploaded_file = st.file_uploader("Upload a WhatsApp .txt file:")
    if uploaded_file:
        content = uploaded_file.read()
        messages = extract_messages_from_whatsapp(content)
        predictions = []

        st.subheader("📱 Message Predictions")
        for msg in messages:
            if len(msg.strip()) > 20:
                label, confidence = predict_news(msg)
                predictions.append(1 if label == "Fake" else 0)
                st.markdown(f"- {msg[:100]}... → **{label}** ({confidence * 100:.1f}%)")

        if predictions:
            plot_prediction_distribution(predictions)
        else:
            st.info("No valid messages found for prediction.")

st.markdown("<br><br><br>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="footer">
        Made with ❤️ by <a href="https://www.linkedin.com/in/shreya-shukla-14638a290/" target="_blank">Shreya Shukla</a>
    </div>
    """,
    unsafe_allow_html=True
)



