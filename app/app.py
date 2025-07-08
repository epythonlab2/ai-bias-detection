# app.py (Main Streamlit App)
import streamlit as st
from modules.logger import Logger
from modules.data_cleaning import (
    load_csv,
    clean_data,
    find_fixable_label_issues,
    handle_label_corrections,
    standardize_labels,
)
from modules.model import load_finbert_model, classify_headlines
from modules.ui import display_results

def main():
    st.set_page_config(page_title="FinBERT Bias Detection", layout="wide")
    st.title("📊 AI Bias Detection App")

    logger = Logger()

    with st.sidebar:
        st.header("📂 Upload Data")
        uploaded_file = st.file_uploader("Upload your labeled CSV file", type=["csv"])
        st.markdown("Upload a CSV with columns: **headline** and **label** (Positive, Negative, Neutral).")
        if st.button("🔄 Reset App"):
            st.session_state.clear()
            st.rerun()

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin.")
        return

    # Load and clean data only once
    if "cleaned_df" not in st.session_state:
        df = load_csv(uploaded_file)
        df = clean_data(df, logger)

        # Rename 'label' column to 'manual_label' immediately
        if 'label' in df.columns:
            df = df.rename(columns={'label': 'manual_label'})

        st.session_state["cleaned_df"] = df
        st.session_state["corrections_submitted"] = False

    df = st.session_state["cleaned_df"]

    # Check for manual_label presence
    if "manual_label" not in df.columns:
        st.error("Data missing 'manual_label' column. Please reset the app and upload a valid file.")
        st.stop()

    # Detect and fix label issues
    fixable_df = find_fixable_label_issues(df.rename(columns={'manual_label': 'label'}))  # adapt function expecting 'label'

    if not fixable_df.empty and not st.session_state["corrections_submitted"]:
        # The handle_label_corrections expects 'label' col, so pass renamed df temporarily
        temp_df, fixed = handle_label_corrections(df.rename(columns={'manual_label': 'label'}), fixable_df, logger)

        if fixed:
            # Rename back to manual_label before saving
            temp_df = temp_df.rename(columns={'label': 'manual_label'})
            st.session_state["cleaned_df"] = temp_df
            st.session_state["corrections_submitted"] = True
            st.rerun()

    elif st.session_state["corrections_submitted"]:
        fixable_df = find_fixable_label_issues(df.rename(columns={'manual_label': 'label'}))
        if fixable_df.empty:
            st.success("✅ All labels are valid and present (corrections submitted).")
        else:
            st.warning("⚠️ Some labels are still invalid. Please correct them.")
            st.session_state["corrections_submitted"] = False
            st.rerun()
    else:
        st.success("✅ All labels are valid and present.")

    # Standardize labels
    df = standardize_labels(df.rename(columns={'manual_label': 'label'}))
    df = df.rename(columns={'label': 'manual_label'})

    # Load model and classify
    classifier = load_finbert_model()
    df = classify_headlines(df, classifier)

    # Display results
    display_results(df)

    with st.expander("Show Data Cleaning Logs"):
        logger.render_logs()


if __name__ == "__main__":
    main()
