import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

# Function to preprocess the input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load the dataset
@st.cache_data
def load_data():
    file_path = r"C:\Users\yp104\Desktop\t\archive\Ecommerce_FAQ_Chatbot_dataset.json"
    df = pd.read_json(file_path)
    df['question'] = df['questions'].apply(lambda x: x.get('question', ''))
    df['answer'] = df['questions'].apply(lambda x: x.get('answer', ''))
    df = df.drop(columns=['questions'])
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    return df

# Function to create TF-IDF model
def build_vectorizer(questions):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix

# Function to get chatbot response
def get_chatbot_response(user_query, vectorizer, tfidf_matrix, faq_df, threshold=0.2):
    user_query = preprocess_text(user_query)
    query_vec = vectorizer.transform([user_query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(sims)

    if sims[best_idx] < threshold:
        return "Sorry, I couldn't find an answer to that. Please try asking something else."
    else:
        return faq_df.loc[best_idx, 'answer']

# Streamlit UI
st.set_page_config(page_title="E-commerce FAQ Chatbot", layout="centered")
st.title("🛍️ E-commerce FAQ Chatbot")
st.write("Ask a question about your e-commerce orders, returns, delivery, and more.")

# Load data and model
faq_df = load_data()
vectorizer, tfidf_matrix = build_vectorizer(faq_df['question'])

# User input
user_input = st.text_input("Enter your question here:", placeholder="e.g., How do I track my order?")

if user_input:
    response = get_chatbot_response(user_input, vectorizer, tfidf_matrix, faq_df)
    st.markdown("**Chatbot:** " + response)
