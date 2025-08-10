import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json

# 1️⃣ Load the model from Hugging Face
MODEL_NAME = "jordan88rali/mental-health-chatbot-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# 2️⃣ Load label map from repo (same as during training)
label_map = {
    0: "greeting", 1: "goodbye", 2: "thanks", 3: "about",
    4: "help", 5: "happy", 6: "sad", 7: "stressed",
    8: "anxious", 9: "depressed", 10: "default", 11: "suicide"
}

# 3️⃣ Streamlit UI
st.title("💬 Mental Health Chatbot")
st.write("Type your message below and I'll detect your intent.")

user_input = st.text_input("You:")

if user_input:
    # Tokenize and predict
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    predicted_intent = label_map[predicted_class_id]
    st.write(f"**Predicted Intent:** {predicted_intent}")
