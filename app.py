import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -------------------------------
# 1. Load Model & Tokenizer
# -------------------------------
MODEL_NAME = "jordan88rali/mental-health-chatbot-v2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# 2. Custom Labels & Responses
# -------------------------------
labels = [
    'greeting', 'goodbye', 'thanks', 'about', 'help', 'happy',
    'sad', 'stressed', 'anxious', 'depressed', 'default', 'suicide'
]

id2label = {i: label for i, label in enumerate(labels)}

# Predefined empathetic responses for each intent
responses = {
    "greeting": "Hello! How are you feeling today?",
    "goodbye": "Take care! Remember, I’m always here to chat.",
    "thanks": "You're welcome! I'm glad I could help.",
    "about": "I'm a mental health support chatbot, here to listen and guide you.",
    "help": "I'm here to help. Please share what’s troubling you.",
    "happy": "That's wonderful! I'm glad you're feeling good today.",
    "sad": "I'm sorry you're feeling sad. I'm here to listen and support you.",
    "stressed": "Stress can be tough. Want to talk about what's causing it?",
    "anxious": "Anxiety can be overwhelming. Let's talk about it.",
    "depressed": "I'm sorry you’re feeling this way. You’re not alone in this.",
    "default": "I’m not sure I understood, but I’m here to chat about anything.",
    "suicide": "If you’re thinking about suicide, please seek help immediately. In the UK, call Samaritans at 116 123."
}

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="💬 Mental Health Chatbot", page_icon="🧠")
st.title("💬 Mental Health Chatbot (V2)")
st.write("Type your message below and I'll respond in a human-like way.")

user_input = st.text_input("You:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()

    predicted_label = id2label[predicted_class_id]
    bot_response = responses.get(predicted_label, "I’m here for you, whatever’s on your mind.")

    st.markdown(f"**Predicted Intent:** {predicted_label}")
    st.markdown(f"**Chatbot:** {bot_response}")

# -------------------------------
# 4. Footer
# -------------------------------
st.markdown("---")
st.caption("⚠️ This chatbot is for support only and not a substitute for professional mental health care.")
