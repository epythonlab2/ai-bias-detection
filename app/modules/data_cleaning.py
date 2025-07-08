# --- modules/data_cleaning.py ---

import pandas as pd
import streamlit as st
from typing import Tuple
from modules.logger import Logger

VALID_LABELS = ["Positive", "Negative", "Neutral"]

def load_csv(file) -> pd.DataFrame:
    """Load CSV file and validate required columns."""
    df = pd.read_csv(file)
    if 'headline' not in df.columns or 'label' not in df.columns:
        st.error("CSV must contain 'headline' and 'label' columns.")
        st.stop()
    return df[['headline', 'label']].copy()

def clean_data(df: pd.DataFrame, logger: Logger) -> pd.DataFrame:
    """Drop rows with missing headlines and log."""
    missing_mask = df['headline'].isna()
    if missing_mask.sum() > 0:
        logger.log(f"Dropped {missing_mask.sum()} rows with missing headlines.")
        st.info(f"Dropped {missing_mask.sum()} rows with missing headlines.")
        df = df[~missing_mask]
    else:
        st.success("✅ No missing headlines found.")
    return df

def find_fixable_label_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Return subset of rows with missing or invalid labels."""
    missing_labels_mask = df['label'].isna()
    invalid_labels_mask = ~df['label'].isin(VALID_LABELS)
    fixable_mask = missing_labels_mask | invalid_labels_mask
    return df[fixable_mask].copy()

def handle_label_corrections(
    df: pd.DataFrame,
    fixable_df: pd.DataFrame,
    logger: Logger,
) -> Tuple[pd.DataFrame, bool]:
    """Prompt user to fix or remove rows with missing/invalid labels."""
    st.warning(f"{len(fixable_df)} rows need label correction or removal.")
    st.dataframe(fixable_df, use_container_width=True)

    fix_option = st.radio(
        "Handle missing or invalid labels:",
        options=["Select an option", "Remove rows", "Manually fix labels"],
        index=0,
    )

    if fix_option == "Select an option":
        st.warning("Please select an option to continue.")
        st.stop()

    elif fix_option == "Remove rows":
        df = df.drop(fixable_df.index)
        logger.log(f"Removed {len(fixable_df)} rows with missing or invalid labels.")
        st.info(f"Removed {len(fixable_df)} rows with missing or invalid labels.")
        return df, True

    elif fix_option == "Manually fix labels":
        corrected_rows = []
        with st.form("label_fix_form"):
            for idx, row in fixable_df.iterrows():
                st.markdown(f"**Row {idx}**: {row['headline']}")
                corrected_label = st.selectbox(
                    f"Fix label for Row {idx}", VALID_LABELS, key=f"fix_{idx}"
                )
                corrected_rows.append((idx, corrected_label))
            submitted = st.form_submit_button("Submit Corrections")

        if submitted:
            for idx, new_label in corrected_rows:
                original = df.at[idx, 'label']
                df.at[idx, 'label'] = new_label
                logger.log(f"Corrected label for Row {idx} from '{original}' to '{new_label}'")
            still_invalid = df['label'].isna() | ~df['label'].isin(VALID_LABELS)
            if still_invalid.any():
                st.error("Some labels are still invalid. Please fix them.")
                st.stop()
            st.success("✅ All label corrections submitted successfully.")
            st.rerun()
        else:
            st.stop()

    else:  # Fallback
        st.stop()

def standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize label text to lowercase."""
    mapping = {l: l.lower() for l in VALID_LABELS}
    df['label'] = df['label'].map(mapping).fillna(df['label'])
    return df
