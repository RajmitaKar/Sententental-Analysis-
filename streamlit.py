import streamlit as st
import joblib
import re

# Load model assets
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit UI
st.set_page_config(page_title="Mood Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Mood Classifier")
st.markdown("Enter a sentence and I'll guess the mood!")

user_input = st.text_area("💬 Your text:")

if st.button("Predict Mood"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)
        mood = label_encoder.inverse_transform(pred)[0]

        mood_emojis = {
            "joy": "😄",
            "sadness": "😢",
            "love": "💖",
            "anger": "😠",
            "fear": "😱"
        }

        st.success(f"**Mood:** {mood_emojis.get(mood, '')} {mood.capitalize()}")
    else:
        st.warning("Please enter some text!")
