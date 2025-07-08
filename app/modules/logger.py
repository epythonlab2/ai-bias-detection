# --- modules/logger.py
from typing import List
import streamlit as st

class Logger:
    def __init__(self):
        if "logs" not in st.session_state:
            st.session_state["logs"] = []

    def log(self, message: str):
        st.session_state["logs"].append(message)

    def get_logs(self) -> List[str]:
        return st.session_state.get("logs", [])

    def clear(self):
        st.session_state["logs"] = []

    def render_logs(self):
        st.header("📄 Data Cleaning Log")
        logs = self.get_logs()
        if logs:
            for msg in logs:
                st.write("- ", msg)
        else:
            st.write("No data cleaning actions performed.")
