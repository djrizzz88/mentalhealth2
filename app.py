import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Load model from Hugging Face ---
MODEL_NAME = "jordan88rali/mental-health-chatbot-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# --- Get labels directly from model config ---
id2label = model.config.id2label

st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")
st.title("💬 Mental Health Chatbot (V2)")

# --- User input ---
user_input = st.text_input("Type your message below and I'll try to understand you:")

if st.button("Send"):
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class_id = outputs.logits.argmax().item()
        predicted_label = id2label[predicted_class_id]  # ✅ FIXED
        
        st.markdown(f"**Predicted Intent:** `{predicted_label}`")
    else:
        st.warning("Please type something before sending.")
