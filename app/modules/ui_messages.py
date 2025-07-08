# --- modules/ui_messages.py

import streamlit as st

def show_file_errors():
    st.error("CSV must contain 'headline' and 'label' columns.")

def show_no_missing_headlines():
    st.success("✅ No missing headlines found.")

def show_missing_labels_info(missing_count, invalid_count):
    st.info(f"Missing Labels: {missing_count} | Invalid Labels: {invalid_count}")

def show_label_correction_warning(rows_to_fix):
    st.warning(f"{rows_to_fix} rows need label correction or removal.")

def show_label_corrections_success():
    st.success("✅ All label corrections submitted successfully.")

def show_all_labels_valid():
    st.success("✅ All labels are valid and present.")

def show_all_labels_valid_after_correction():
    st.success("✅ All labels are valid and present (corrections submitted).")

def show_loading_model():
    st.info("Loading FinBERT model...")

def show_classifying_headlines():
    st.info("Classifying headlines...")

def show_no_data_uploaded():
    st.info("Please upload a CSV file to begin.")

def show_invalid_corrections_remaining():
    st.error("Some labels are still invalid. Please fix them.")
