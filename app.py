import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import random

# -------------------------
# 1️⃣ Load the model + tokenizer from Hugging Face
# -------------------------
MODEL_NAME = "jordan88rali/mental-health-chatbot-v2"  # your model repo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# -------------------------
# 2️⃣ Load intents file (responses)
# -------------------------
with open("simple_intents.json", "r") as f:
    intents = json.load(f)["intents"]

# Map intents to responses
intent_responses = {intent["tag"]: intent["responses"] for intent in intents}

# Get labels from config (label_map.json)
with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Reverse label map to get label name from id
id2label = {v: k for k, v in label_map.items()}

# -------------------------
# 3️⃣ Chatbot function
# -------------------------
def predict_intent(user_text):
    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
    intent_label = id2label[predicted_class_id]
    return intent_label

def get_bot_response(intent_label):
    if intent_label in intent_responses:
        return random.choice(intent_responses[intent_label])
    else:
        return "I'm here for you. Could you tell me more about what’s on your mind?"

# -------------------------
# 4️⃣ Streamlit UI
# -------------------------
st.title("💬 Mental Health Chatbot")
st.write("I’m here to listen. How are you feeling today?")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip():
    intent = predict_intent(user_input)
    bot_reply = get_bot_response(intent)

    # Save conversation
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_reply))

# Show chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")
