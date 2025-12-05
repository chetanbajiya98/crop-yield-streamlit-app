import streamlit as st
import joblib
import requests
import os

MODEL_URL = "https://huggingface.co/spaces/chetanbajiya/crop-yield-api/resolve/main/yield_model5.pkl"

@st.cache_resource
def load_model():
    local_path = "yield_model5.pkl"

    # Download the model only if not exists
    if not os.path.exists(local_path):
        st.write("⬇️ Downloading ML model from Hugging Face...")
        response = requests.get(MODEL_URL)

        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
        else:
            st.error("❌ Model download failed.")
            st.stop()

    try:
        model = joblib.load(local_path)
        return model
    except Exception as e:
        st.error("❌ Model loading failed. Version mismatch or corrupted file.")
        st.write(e)
        st.stop()

model = load_model()
