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

st.set_page_config(page_title="🍽️ Zomato Restaurant Chatbot", layout="wide")

# Google Drive file ID and names
ZIP_FILE_ID = "1cTIdyYhkSBGvk5TSqWiBOiORakFDWpzV"  # Corrected file ID
ZIP_FILE_NAME = "zomato_bot_package.zip"
EXTRACT_PATH = "zomato_bot_package"

# Download and extract ZIP if not already done
if not os.path.exists(EXTRACT_PATH):
    st.write("Downloading model/data zip from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={ZIP_FILE_ID}", ZIP_FILE_NAME, quiet=False)

    st.write("Extracting zip...")
    with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

    st.write("Cleaning up...")
    os.remove(ZIP_FILE_NAME)

# Define the path for the resources
DATA_PATH = 'zomato_bot_package'

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
    return df, index, embedder, sentiment_pipeline

df, index, embedder, sentiment_pipeline = load_resources()

# Tag extraction from dish_liked
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

# Input box
st.title("🍽️ Zomato Restaurant Chatbot")
query = st.text_input("What are you craving today?")

if query:
    q_embedding = embedder.encode([query])
    D, I = index.search(np.array(q_embedding), k=30)

    top_df = df.iloc[I[0]].drop_duplicates(subset="name")
    top_df = top_df.sort_values(by="rate", ascending=False).head(10)

    if top_df.empty:
        st.warning("🥲 No matches found. Try a different query.")
    else:
        st.subheader(f"🍴 Top {len(top_df)} Restaurant Matches")

        # Prepare map data if lat/lon exist
        if {'latitude', 'longitude'}.issubset(top_df.columns):
            st.map(top_df[['latitude', 'longitude']].dropna())

        for _, row in top_df.iterrows():
            sentiment_score = row.get("sentiment_score", 0)
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
            - 🧠 Sentiment Score: {sentiment_score}
            - 🏷️ Tags: {tags}
            - 💬 Reviews: {str(row['reviews_list'])[:300]}...
            - 🔗 Zomato Page: {zomato_link}
            - 🗺️ [View on Map]({maps_link})
            ---
            """)

        st.success("Scroll above to view recommended restaurants 👆")
        st.markdown("Craving something else? Ask me again!")
