
# 🧠 Bias Detection in Financial Headlines using FinBERT

This project uses **FinBERT** to detect potential sentiment bias in financial headlines. It includes a full-featured **Streamlit app** for visual inspection, evaluation, export of misclassified samples, and optional **retraining**.

---

## 📌 Project Overview

| Step | Description |
|------|-------------|
| 1. Define Bias | Identify bias from **excessively positive/negative framing** in finance |
| 2. Collect Data | Scrape or collect 100–300 real-world financial headlines |
| 3. Label Manually | Tag each headline as `Positive`, `Neutral`, or `Negative` |
| 4. Classify | Run FinBERT zero-shot sentiment classification |
| 5. Evaluate | Analyze model performance, confidence, confusion matrix |
| 6. Retrain | Export misclassified samples for optional fine-tuning |
| 7. Visualize | Use a Streamlit UI to interactively explore and export results |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/epythonlab/ai-bias-detection.git
cd ai-bias-detection
```

### 2. Create and Activate Environment

Using Conda:

```bash
conda create -n biasdetect python=3.10
conda activate biasdetect
```

Or using venv:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

Launch the Streamlit app using the `starter.py` script:

```bash
python starter.py
```

This automatically runs:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
├── starter.py                    # Starts the Streamlit app
├── app/app.py                        # Main app logic (UI, routing)
├── requirements.txt              # Dependencies
├── data/                         # Input/output datasets
├── notebook/                     # Optional notebook version
│   └── Financial_Headline_Bias_Classification.ipynb
├── app/modules/
│   ├── ui.py                     # UI rendering, visualizations, download buttons
│   ├── ui_messages.py            # Centralized status/info messages
│   ├── model.py                  # FinBERT loading and classification
│   ├── retraining.py             # Finetune FinBERT using misclassified samples
│   ├── logger.py                 # Logs for data cleaning and label issues
│   └── data_cleaning.py          # Cleans, validates, and fixes label inconsistencies
└── README.md
```

---

## 🧪 Input Format

Upload a CSV with:

| Column    | Description                             |
|-----------|-----------------------------------------|
| headline  | Financial news headline text            |
| label     | Manually assigned label: Positive, etc. |

---

## 📊 Output Columns

After FinBERT classification:

| Column              | Description                          |
|---------------------|--------------------------------------|
| `headline`          | Original headline                    |
| `manual_label`      | Cleaned human-assigned label         |
| `finbert_label`     | Label predicted by FinBERT           |
| `finbert_confidence` | Prediction confidence (0–1)        |

---

## 🔁 Retraining with Misclassified Samples

From the app, you can:

- View & export misclassified examples
- Add them to a retrain dataset (`retrain_dataset.csv`)
- Use `modules/retraining.py` to finetune FinBERT with that dataset

### Example:

```bash
python -m modules.retraining --input data/retrain_dataset.csv
```

---

## ✅ Requirements

```
streamlit
transformers
torch
pandas
scikit-learn
plotly
datasets
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- FinBERT is used in zero-shot mode (no fine-tuning required to start)
- Exported data can help create a domain-adapted version of the model
- Clean UI messages are managed in `ui_messages.py`

---

## 📧 Contact

Have a question, idea, or feedback?  
Open an issue or email **asibeh.tenager@gmail.com**.
