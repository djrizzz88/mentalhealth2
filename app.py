import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================================
# 1️⃣ Load model & tokenizer
# ================================
MODEL_NAME = "jordan88rali/mental-health-chatbot-v2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# ================================
# 2️⃣ Load label map & fix format
# ================================
with open("label_map.json", "r") as f:
    label_map = json.load(f)

if isinstance(label_map, list):
    id2label = {i: label for i, label in enumerate(label_map)}
else:
    id2label = {v: k for k, v in label_map.items()}

# ================================
# 3️⃣ Streamlit UI setup
# ================================
st.set_page_config(page_title="🧠 Mental Health Chatbot", page_icon="💬")
st.title("🧠 Mental Health Support Chatbot")
st.markdown("Hello 👋. I’m here to talk and listen. How are you feeling today?")

# Store conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================================
# 4️⃣ Predict intent function
# ================================
def predict_intent(user_text):
    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    intent = id2label[prediction]
    return intent

# ================================
# 5️⃣ Chat interaction
# ================================
if prompt := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Predict intent
    intent = predict_intent(prompt)

    # Generate a human-like bot reply
    if intent == "greeting":
        bot_reply = "Hello! How are you feeling today? 🌼"
    elif intent == "sad":
        bot_reply = "I’m sorry you’re feeling this way. Do you want to talk more about it? 💙"
    elif intent == "happy":
        bot_reply = "That’s wonderful to hear! 🌟 What’s been going well for you?"
    elif intent == "stressed":
        bot_reply = "I understand stress can be tough. Want to share what’s on your mind? 🤝"
    elif intent == "suicide":
        bot_reply = "I’m really concerned about your safety. You are not alone — please call a suicide helpline immediately. 📞"
    else:
        bot_reply = f"I think you might be feeling **{intent}**. Would you like to tell me more?"

    # Add bot reply to history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
