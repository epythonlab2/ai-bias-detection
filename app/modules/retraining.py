# --- modules/retraining.py ---

import os
import time
import zipfile
import pandas as pd
import torch
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

# 🔧 Patch accelerate.unwrap_model to bypass known bug
from accelerate import Accelerator

original_unwrap_model = Accelerator.unwrap_model

def patched_unwrap_model(self, model, **kwargs):
    return original_unwrap_model(self, model)

Accelerator.unwrap_model = patched_unwrap_model

MODEL_NAME = "yiyanghkust/finbert-tone"
OUTPUT_DIR = "finbert-finetuned"
EXPORT_ZIP = "finbert-finetuned.zip"

label2id = {"positive": 0, "neutral": 1, "negative": 2}
id2label = {v: k for k, v in label2id.items()}

torch.set_num_threads(2)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def preprocess_data(df: pd.DataFrame):
    df = df.rename(columns={"headline": "text", "manual_label": "label"})
    df = df[df["label"].str.lower().isin(label2id)]
    df["label"] = df["label"].str.lower().map(label2id)
    return Dataset.from_pandas(df)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    accuracy = (predictions == labels).astype(float).mean()
    return {"accuracy": accuracy}

def zip_model(output_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for foldername, _, filenames in os.walk(output_dir):
            for filename in filenames:
                filepath = os.path.join(foldername, filename)
                arcname = os.path.relpath(filepath, output_dir)
                zipf.write(filepath, arcname)

def finetune_model(df: pd.DataFrame, callback=None):
    logs = []
    start_time = time.time()
    try:
        def log(msg):
            logs.append(msg)
            if callback:
                callback(msg)

        log("📦 Preprocessing data...")
        dataset = preprocess_data(df)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        log("🤖 Loading FinBERT model...")
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=id2label,
            label2id=label2id
        )

        use_cpu = not torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to="none",

            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,

            use_cpu=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            disable_tqdm=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        log("🟢 Starting training...")
        trainer.train()

        log("💾 Saving fine-tuned model...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        log("🗘️ Zipping model...")
        zip_model(OUTPUT_DIR, EXPORT_ZIP)

        elapsed = time.time() - start_time
        log(f"✅ Model saved and zipped to {EXPORT_ZIP} in {elapsed:.2f} seconds.")

    except Exception as e:
        log(f"❌ Error during fine-tuning: {str(e)}")

    return "\n".join(logs)

def finetune_and_predict(df_original: pd.DataFrame, callback=None):
    logs = finetune_model(df_original, callback=callback)

    model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR)
    tokenizer_loaded = BertTokenizerFast.from_pretrained(OUTPUT_DIR)

    def classify(texts):
        inputs = tokenizer_loaded(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=1).numpy()
        confs = torch.nn.functional.softmax(logits, dim=1).max(dim=1).values.numpy()
        return preds, confs

    texts = df_original["headline"].tolist()
    preds, confs = classify(texts)
    id2label_local = {v: k for k, v in label2id.items()}

    df_new = df_original.copy()
    df_new["finbert_label"] = [id2label_local[p] for p in preds]
    df_new["finbert_confidence"] = confs
    df_new["correct"] = df_new["manual_label"] == df_new["finbert_label"]

    return df_new, logs

