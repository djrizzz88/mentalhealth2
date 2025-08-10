import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random

# -------------------
# Load Model & Tokenizer
# -------------------
MODEL_NAME = "jordan88rali/mental-health-chatbot-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# -------------------
# Map model labels to intent names
# -------------------
id2label = model.config.id2label
label2id = model.config.label2id

# -------------------
# Multiple responses for each intent
# -------------------
intent_responses = {
    "greeting": [
        "Hello! How are you feeling today?",
        "Hi there — what’s on your mind?",
        "Hey! Tell me how you're doing today.",
        "Hi! How are things going for you?"
    ],
    "goodbye": [
        "Bye! Take care of yourself.",
        "Talk soon — I’m here whenever you need.",
        "See you later. Be kind to yourself.",
        "Goodbye for now, remember you’re not alone."
    ],
    "thanks": [
        "You’re welcome! I’m glad it helped.",
        "Any time — I’m here for you.",
        "Happy to help.",
        "No problem at all!"
    ],
    "sad": [
        "I’m sorry you’re feeling this way. Want to share more about what’s going on?",
        "That sounds tough. I’m here to listen.",
        "I understand—it’s okay to feel this way sometimes.",
        "I’m here for you, no matter what."
    ],
    "stressed": [
        "That sounds stressful. Do you want to talk through what’s causing it?",
        "I hear you—stress can be overwhelming. Let’s take it one step at a time.",
        "It’s okay to take a break and breathe.",
        "I understand—stress can feel heavy. You’re not alone."
    ],
    "anxious": [
        "Anxiety can be exhausting. I’m here with you.",
        "It’s okay to feel anxious. Do you want to share what’s on your mind?",
        "You’re safe here. Let’s talk about it.",
        "I understand—anxiety can feel intense, but it will pass."
    ],
    "suicide": [
        "I’m really concerned about your safety. You matter a lot to me. Please call a suicide helpline immediately.",
        "Your life is important. Please reach out to a suicide prevention line now.",
        "I hear you’re in deep pain. Please talk to someone right away at a helpline.",
        "You’re not alone—help is available right now. Please contact a crisis line."
    ],
    "default": [
        "I’m not sure I understood that. Could you say it another way?",
        "Hmm, I’m not sure what you mean. Can you rephrase?",
        "I want to understand—could you explain more?",
        "Let’s try that again—can you tell me in a different way?"
    ]
}

# -------------------
# Keyword override for small dataset accuracy
# -------------------
keyword_overrides = {
    "sad": ["sad", "unhappy", "down", "depressed"],
    "stressed": ["stress", "overwhelmed", "pressure", "burnout"],
    "anxious": ["anxious", "nervous", "worried", "panicking"],
    "suicide": ["suicide", "kill myself", "end my life", "can't go on"]
}

def apply_keyword_override(user_input, predicted_label):
    text = user_input.lower()
    for label, keywords in keyword_overrides.items():
        if any(k in text for k in keywords):
            return label
    return predicted_label

# -------------------
# Prediction Function
# -------------------
def predict_intent(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits).item()
    predicted_label = id2label[predicted_class_id]

    # Apply keyword override
    predicted_label = apply_keyword_override(user_input, predicted_label)
    return predicted_label

# -------------------
# Streamlit UI
# -------------------
st.title("💬 Mental Health Chatbot (V2)")
st.write("Type your message below and I'll respond in a human-like way.")

user_input = st.text_input("You:")

if user_input:
    intent = predict_intent(user_input)
    response = random.choice(intent_responses.get(intent, intent_responses["default"]))

    st.write(f"**Predicted Intent:** {intent}")
    st.write(f"**Chatbot:** {response}")
