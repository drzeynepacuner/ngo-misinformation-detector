import os
import streamlit as st
import torch
import numpy as np

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from lime.lime_text import LimeTextExplainer

# === Load model and tokenizer from Hugging Face ===
model = DistilBertForSequenceClassification.from_pretrained(
    "doomzoom/distilbert-misinformation",
    use_safetensors=True
)
tokenizer = DistilBertTokenizerFast.from_pretrained("doomzoom/distilbert-misinformation")

# === LIME setup ===
class_names = ["Fake/Misleading", "Real/Trustworthy"]


def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.numpy()


explainer = LimeTextExplainer(class_names=class_names)

# === Streamlit UI ===
st.set_page_config(page_title="NGO Misinformation Risk Detector", layout="centered")
st.title("🧠 NGO Misinformation Risk Detector")
st.markdown(
    "Paste a message or article below. This tool will predict how likely it is to be misleading — and explain why.")

text_input = st.text_area("📝 Paste your content here:", height=200)

if st.button("🔍 Analyze"):
    if text_input.strip() == "":
        st.warning("Please paste some text to analyze.")
    else:
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = float(probabilities[0][predicted_class])

        label = class_names[predicted_class]
        emoji = "✅" if predicted_class == 1 else "⚠️"

        st.markdown(f"### {emoji} Prediction: **{label}**")
        st.markdown(f"Confidence Score: `{confidence:.2f}`")

        st.markdown("#### 🧠 Why this result?")
        explanation = explainer.explain_instance(text_input, predict_proba, num_features=8, labels=[predicted_class])

        # Style override for dark mode
        lime_html = explanation.as_html()
        custom_style = """
        <style>
            body, html {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            table td, table th {
                color: #ffffff !important;
                background-color: #2e2e2e !important;
            }
            .lime, .lime .word, .lime .score {
                color: #ffffff !important;
                background-color: #2e2e2e !important;
            }
            a, a:visited {
                color: #ccccff !important;
            }
        </style>
        """
        st.components.v1.html(custom_style + lime_html, height=400, scrolling=True)
