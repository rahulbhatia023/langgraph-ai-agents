import tempfile

import streamlit as st

from agents.data_query_assistant_agent import DataQueryAssistantAgent
from common.page import BasePage


class DataQueryAssistantPage(BasePage):
    agent = DataQueryAssistantAgent

    required_keys = ["OPENAI_API_KEY"]


DataQueryAssistantPage.display()

with st.sidebar:
    st.markdown(
        "<br/><br/><h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>Upload SQLite DB file</h3>",
        unsafe_allow_html=True,
    )

    if uploaded_file := st.file_uploader(
        label="Upload SQLite DB file", type=["csv", "sqlite"], label_visibility="hidden"
    ):
        with tempfile.NamedTemporaryFile(delete=False) as sqlite_file:
            sqlite_file.write(uploaded_file.read())
            sqlite_file.flush()
            st.session_state["uploaded_file"][
                DataQueryAssistantAgent.name
            ] = sqlite_file.name
