import streamlit as st
import numpy as np
import pickle
import emoji
import re
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tensorflow.keras.models import load_model
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

#import streamlit as st
#import gdown
#import os

# Google Drive links to the files
#file_1_url = "https://drive.google.com/uc?id=1gjn785gREju8bK5VbJhwJZK-oezQurMa"
#file_2_url = "https://drive.google.com/uc?id=1JPQmbfl9nGDDUqKoHtPGPOyJyDhat9pb"

# Paths to where you want to store the files
#file_1_path = "model_components/libtensorflow_cc.so.2"
#file_2_path = "model_components/libtensorflow_framework.so.2"

# Function to download the files if they don't exist
#def download_files():
#    if not os.path.exists(file_1_path) or not os.path.exists(file_2_path):
#        st.info("Downloading necessary files...")
#        gdown.download(file_1_url, file_1_path, quiet=False)
#        gdown.download(file_2_url, file_2_path, quiet=False)
#        st.success("Files downloaded successfully!")

# Download files at the start
#download_files()


# Load artifacts
model = load_model("model_components/emotion_nn_model.h5")
with open("model_components/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model_components/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Sentiment Mapping
sentiment_mapping = {
    "sad": "Negative",
    "fear": "Negative",
    "anger": "Negative",
    "love": "Positive",
    "joy": "Positive",
    "suprise": "Neutral"
}

negation_words = {"not", "no", "never", "none", "nothing", "nobody", "neither", "nowhere", "without"}

def get_antonym(word):
    antonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name().lower())
    return next(iter(antonyms), None)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text)
    text = text.replace(":", " ").lower()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    processed_words = []
    negate = False

    for word in words:
        if word in negation_words:
            negate = True
            continue
        if negate:
            antonym = get_antonym(word)
            if antonym and antonym not in ENGLISH_STOP_WORDS:
                processed_words.append(antonym)
            else:
                processed_words.append(f"neg_{word}")
            negate = False
        else:
            if word not in ENGLISH_STOP_WORDS:
                processed_words.append(word)

    return ' '.join(processed_words)

def predict_emotion(text):
    clean_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([clean_text]).toarray()
    prediction = model.predict(vectorized_text)[0]
    predicted_index = np.argmax(prediction)
    predicted_emotion = label_encoder.classes_[predicted_index]
    predicted_sentiment = sentiment_mapping.get(predicted_emotion.lower(), "Neutral")
    return predicted_emotion.capitalize(), predicted_sentiment, prediction

# Custom CSS for background image
st.markdown(
    """
    <style>
    body {
        background-image: url("file:///images/cheems-z7bq2c62esomoun6.jpg");
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
    }
    </style>
    """, 
    unsafe_allow_html=True
)


# Streamlit UI
st.title("Emotion & Sentiment Analyzer (From the frustrated researcher )")

user_input = st.text_area("Enter a sentence to analyze:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        emotion, sentiment, probs = predict_emotion(user_input)

        st.subheader("🔍 Results")
        st.write(f"**Predicted Emotion:** {emotion}")
        st.write(f"**Mapped Sentiment:** {sentiment}")

        st.subheader("Emotion Probabilities")
        for emo, prob in zip(label_encoder.classes_, probs):
            st.write(f"**{emo.capitalize()}**: {prob*100:.2f}%")
