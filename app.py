import streamlit as st
import requests
import os
import joblib  # <-- IMPORTANT

MODEL_URL = "https://huggingface.co/spaces/chetanbajiya/crop-yield-api/resolve/main/yield_model5.pkl"
MODEL_PATH = "yield_model5.pkl"

@st.cache_resource
def load_model():
    # Download if not present
    if not os.path.exists(MODEL_PATH):
        st.write("⬇️ Downloading ML model from Hugging Face...")
        r = requests.get(MODEL_URL)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        else:
            st.error("❌ Failed to download model.")
            st.stop()

    # Try loading the model
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error("❌ Model loading failed.")
        st.write(e)
        st.stop()

