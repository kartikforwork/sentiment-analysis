import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Set page config for better UI
st.set_page_config(
    page_title="TrendSent",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautification with improved text visibility
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextArea>div>div>textarea {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            color: #000000 !important;  /* Ensures text is black */
        }
        .stTextArea>div>div>textarea::placeholder {
            color: #6c757d;  /* Gray placeholder text */
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px 24px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .title {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .positive {
            background-color: #d4edda;
            color: #155724;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #28a745;
        }
        .negative {
            background-color: #f8d7da;
            color: #721c24;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #dc3545;
        }
    </style>
""", unsafe_allow_html=True)

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Function to create a colored card
def create_card(text, sentiment):
    if sentiment == "Positive":
        st.markdown(f'<div class="positive"><b>😊 Positive Sentiment</b><br>{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="negative"><b>😞 Negative Sentiment</b><br>{text}</div>', unsafe_allow_html=True)

# Main app logic
def main():
    st.markdown("""
        <style>
            .main-title {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
                font-weight: bold;
            }
            .sub-title {
                color: #5d6d7e;
                text-align: center;
                margin-bottom: 30px;
                font-size: 1.2em;
                font-style: italic;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">TrendSent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Predicting Market Trends from Twitter Sentiments</div>', unsafe_allow_html=True)
    st.markdown("Analyze the sentiment of your text to understand if it's positive or negative.")

    # Load stopwords, model, and vectorizer
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    # User input with improved visibility
    text_input = st.text_area(
        "Paste text that you want to analyze",
        placeholder="Paste tweet or any text that you want to analyze for sentiment...",
        height=150,
        key="text_input"  # Added key for better component identification
    )
    
    if st.button("Analyze Sentiment"):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Analyzing sentiment..."):
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                create_card(text_input, sentiment)

if __name__ == "__main__":
    main()