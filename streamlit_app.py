import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import os
import gdown
import zipfile

st.set_page_config(page_title="🍽️ Zomato GPT Chatbot", layout="wide")

ZIP_FILE_ID = "1cTIdyYhkSBGvk5TSqWiBOiORakFDWpzV"
ZIP_FILE_NAME = "zomato_bot_package.zip"
EXTRACT_PATH = "zomato_bot_package"

if not os.path.exists(EXTRACT_PATH):
    st.write("📦 Downloading model/data...")
    gdown.download(f"https://drive.google.com/uc?id={ZIP_FILE_ID}", ZIP_FILE_NAME, quiet=False)
    with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    os.remove(ZIP_FILE_NAME)

DATA_PATH = EXTRACT_PATH

@st.cache_resource
def load_resources():
    df = pd.read_csv(f'{DATA_PATH}/zomato_clean.csv')
    index = faiss.read_index(f'{DATA_PATH}/zomato_index.faiss')
    embedder = SentenceTransformer(f'{DATA_PATH}/embedding_model')
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    generator = pipeline("text-generation", model="gpt2")
    classifier = pipeline("zero-shot-classification", device=0 if torch.cuda.is_available() else -1)
    return df, index, embedder, sentiment_pipeline, generator, classifier

df, index, embedder, sentiment_pipeline, generator, classifier = load_resources()

def extract_tags(dish_text):
    dish_text = str(dish_text).lower()
    tags = []
    if any(w in dish_text for w in ["veg", "paneer", "aloo", "dal"]):
        tags.append("🥦 Veg")
    if any(w in dish_text for w in ["chicken", "kebab", "tikka"]):
        tags.append("🍗 Chicken")
    if any(w in dish_text for w in ["mutton", "lamb"]):
        tags.append("🥩 Mutton")
    if any(w in dish_text for w in ["spicy", "masala", "mirchi", "tandoori"]):
        tags.append("🌶️ Spicy")
    if not tags:
        tags.append("✨ Popular")
    return ", ".join(tags)

def generate_response(user_query, top_df, history=[]):
    context = "\n".join([
        f"{row['name']} ({row['rate']}⭐) - {row['cuisines']}, at {row['location']}"
        for _, row in top_df.iterrows()
    ][:5])

    history_text = "\n".join([f"User: {q}\nBot: {a}" for q, a in history[-3:]])

    prompt = f"""
You are a friendly restaurant chatbot that helps users find the best places to eat based on their queries.

{history_text}
User: {user_query}
Bot: Based on your query, here are the best places I found: {context}. Choose from these options.
"""

    input_tokens = len(prompt.split())
    max_tokens = 300
    max_new_tokens = max_tokens - input_tokens

    if max_new_tokens <= 0:
        return "Sorry, your input is too long for generating a response. Please try again with a shorter query."

    output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)[0]['generated_text']
    answer = output.split("Bot:")[-1].strip().split("User:")[0].strip()

    return answer

def detect_intent_llm(user_input):
    labels = ["greeting", "restaurant_suggestion", "casual_chat"]
    result = classifier(user_input, labels)
    return result['labels'][0]  # predicted intent

st.title("🍽️ Zomato GPT Chatbot")

# Initializing chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Use st.form to control when the input box and button show up
with st.form(key="user_input_form", clear_on_submit=True):
    query = st.text_input("What are you craving today?")
    submit_button = st.form_submit_button(label="Send")

top_df = pd.DataFrame()

if query and submit_button:  # Ensures only processing happens when the form is submitted
    intent = detect_intent_llm(query)

    if intent == "greeting":
        response = "Hello! 👋 How can I help you today?"
        st.session_state.chat_history.append((query, response))

    elif intent == "restaurant_suggestion":
        q_embedding = embedder.encode([query])
        D, I = index.search(np.array(q_embedding), k=30)

        top_df = df.iloc[I[0]].drop_duplicates(subset="name")
        top_df = top_df.sort_values(by="rate", ascending=False).head(10)

        if top_df.empty:
            st.warning("🥲 Couldn't find matching restaurants. Try again?")
        else:
            chatbot_reply = generate_response(query, top_df, st.session_state.chat_history)
            st.session_state.chat_history.append((query, chatbot_reply))  # Append here

    else:
        response = "I'm happy to chat! 😊 Want food recos or just saying hi?"
        st.session_state.chat_history.append((query, response))

# Display chat history
if st.session_state.chat_history:
    st.subheader("🧠 Chat History")
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

if query and not top_df.empty:
    with st.expander("📋 View Recommended Restaurants"):
        if 'latitude' in top_df.columns and 'longitude' in top_df.columns:
            st.map(top_df[['latitude', 'longitude']].dropna())

        for _, row in top_df.iterrows():
            tags = extract_tags(row.get("dish_liked", ""))
            address = str(row.get("address", row.get("location", "")))
            maps_link = f"https://www.google.com/maps/search/?api=1&query={address.replace(' ', '+')}"
            zomato_link = (
                f"[Visit Link]({row['url']})"
                if pd.notna(row.get('url')) and str(row['url']).startswith("http")
                else "Not Available"
            )

            st.markdown(f"""
            ### {row['name']}
            - 📍 Location: {row['location']}
            - 🍽️ Cuisine: {row['cuisines']}
            - ⭐ Rating: {row['rate']}
            - 🧠 Sentiment Score: {row.get("sentiment_score", 0)}
            - 🏷️ Tags: {tags}
            - 💬 Reviews: {str(row['reviews_list'])[:300]}...
            - 🔗 Zomato Page: {zomato_link}
            - 🗺️ [View on Map]({maps_link})
            ---
            """)
