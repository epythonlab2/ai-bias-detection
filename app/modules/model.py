# --- modules/model.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import streamlit as st

def load_finbert_model() -> pipeline:
    with st.spinner("Loading FinBERT model..."):
        device = 0 if torch.cuda.is_available() else -1
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=3,
            function_to_apply="softmax",
            device= device,
        )

    st.success("✅ FinBERT model loaded successfully.")  # Show loaded confirmation

    return classifier

def classify_headlines(df: pd.DataFrame, classifier: pipeline) -> pd.DataFrame:
    with st.spinner("Classifying headlines..."):
        texts = df['headline'].astype(str).tolist()
        results = classifier(texts)
        df["finbert_label"] = [max(r, key=lambda x: x["score"])["label"] for r in results]
        df["finbert_confidence"] = [max(r, key=lambda x: x["score"])["score"] for r in results]
        df.rename(columns={"label": "manual_label"}, inplace=True)

        df["headline"] = df["headline"].astype(str).str.strip().str.replace("\n", " ").str.replace("  ", " ")
        df["manual_label"] = df["manual_label"].str.title()
        df["finbert_label"] = df["finbert_label"].str.title()
    return df
