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

MODEL_NAME = "yiyanghkust/finbert-tone"
OUTPUT_DIR = "finbert-finetuned"
EXPORT_ZIP = "finbert-finetuned.zip"

label2id = {"positive": 0, "neutral": 1, "negative": 2}
id2label = {v: k for k, v in label2id.items()}

# Limit CPU usage (helps prevent freezing)
torch.set_num_threads(2)

# Load tokenizer once
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


def zip_model(output_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for foldername, _, filenames in os.walk(output_dir):
            for filename in filenames:
                filepath = os.path.join(foldername, filename)
                arcname = os.path.relpath(filepath, output_dir)
                zipf.write(filepath, arcname)


def finetune_model(df: pd.DataFrame):
    logs = []
    start_time = time.time()
    try:
        logs.append("📦 Preprocessing data...")
        dataset = preprocess_data(df)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        logs.append("🤖 Loading FinBERT model...")
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=id2label,
            label2id=label2id
        )

        # 🧠 Auto detect if CUDA available
        use_cpu = not torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,        # 🔻 Smallest possible
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=10,
            eval_strategy="epoch",             # ❌ Disable eval
            save_strategy="no",                   # ❌ Disable saving
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        logs.append("🟢 Starting training...")
        trainer.train()

        logs.append("💾 Saving fine-tuned model...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        logs.append("🗜️ Zipping model...")
        zip_model(OUTPUT_DIR, EXPORT_ZIP)

        elapsed = time.time() - start_time
        logs.append(f"✅ Model saved and zipped to {EXPORT_ZIP} in {elapsed:.2f} seconds.")

    except Exception as e:
        logs.append(f"❌ Error during fine-tuning: {str(e)}")

    return "\n".join(logs)

