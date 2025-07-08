# --- modules/ui.py ---

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from modules.retraining import finetune_model  # <-- Add retraining module import

def display_results(df: pd.DataFrame):
    labels = ["Positive", "Neutral", "Negative"]
    df["correct"] = df["manual_label"] == df["finbert_label"]

    correct_df = df[df["correct"]].copy()
    incorrect_df = df[~df["correct"]].copy()

    tab_metrics, tab_data, tab_logs, tab_finetune = st.tabs([
        "📊 Result Metrics", "📝 Results & Misclassified", "📄 Logs", "🔁 Fine-Tune Model"
    ])

    # === METRICS TAB ===
    with tab_metrics:
        st.markdown("## 📈 Classification Results & Metrics")
        total = len(df)
        correct = df['correct'].sum()
        incorrect = total - correct

        col1, col2, col3 = st.columns(3)
        col1.metric("Total samples", total)
        col2.metric("Correct predictions", correct)
        col3.metric("Incorrect predictions", incorrect)

        acc = accuracy_score(df["manual_label"], df["finbert_label"])
        st.metric("Accuracy", f"{acc:.2f}")

        st.text(classification_report(df["manual_label"], df["finbert_label"]))

        cm = confusion_matrix(df["manual_label"], df["finbert_label"], labels=labels)
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
        )
        fig_cm.update_layout(title_text="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
        st.plotly_chart(fig_cm, use_container_width=True)

        class_counts = pd.DataFrame({
            "True": df["manual_label"].value_counts(),
            "Predicted": df["finbert_label"].value_counts()
        }).fillna(0).astype(int).reset_index().rename(columns={"index": "Label"})

        fig_bar = px.bar(class_counts, x="Label", y=["True", "Predicted"], barmode="group",
                         title="Label Distribution: True vs Predicted")
        st.plotly_chart(fig_bar, use_container_width=True)

        if "finbert_confidence" in df.columns:
            fig_conf = px.histogram(df, x="finbert_confidence", nbins=10, title="Prediction Confidence Distribution",
                                    marginal="rug", histnorm='probability')
            st.plotly_chart(fig_conf, use_container_width=True)

    # === RESULTS TAB ===
    with tab_data:
        st.markdown("## 📝 Results & Misclassified Samples")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"🟢 Correctly Classified ({len(correct_df)})")
            st.dataframe(
                correct_df[["headline", "manual_label", "finbert_label", "finbert_confidence"]].head(10),
                use_container_width=True
            )

        with col2:
            st.subheader(f"🔴 Misclassified ({len(incorrect_df)})")
            st.dataframe(
                incorrect_df[["headline", "manual_label", "finbert_label", "finbert_confidence"]].head(10),
                use_container_width=True
            )

            # Add button to save misclassified samples for retraining
            if st.button("➕ Add Misclassified to Retrain Dataset"):
                retrain_samples = incorrect_df[["headline", "manual_label"]].copy()

                if "retrain_samples" in st.session_state:
                    st.session_state["retrain_samples"] = pd.concat(
                        [st.session_state["retrain_samples"], retrain_samples]
                    ).drop_duplicates().reset_index(drop=True)
                else:
                    st.session_state["retrain_samples"] = retrain_samples

                st.success(f"Added {len(retrain_samples)} misclassified samples for retraining.")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Correct Predictions", correct_df.to_csv(index=False), "correct_predictions.csv")
        with col2:
            st.download_button("📥 Download Misclassified Predictions", incorrect_df.to_csv(index=False), "misclassified_predictions.csv")

    # === LOGS TAB (Optional placeholder) ===
    with tab_logs:
        st.markdown("### Logs will appear here.")

    # === FINETUNE TAB ===
    with tab_finetune:
        st.markdown("## 🔁 Fine-tune FinBERT with New Labels")

        if "retrain_samples" not in st.session_state or st.session_state["retrain_samples"].empty:
            st.info("No retrain data available. Add misclassified samples first.")
        else:
            st.write(f"{len(st.session_state['retrain_samples'])} samples ready for fine-tuning.")
            if st.button("🚀 Start Fine-tuning"):
                with st.spinner("Fine-tuning in progress..."):
                    logs = finetune_model(st.session_state["retrain_samples"])
                    st.success("✅ Fine-tuning complete.")
                    st.text(logs or "No logs returned.")
