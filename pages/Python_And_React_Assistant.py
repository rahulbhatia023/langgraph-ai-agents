import os
import time

import streamlit as st
import streamlit.components.v1 as components

from agents.python_and_react_assistant_agent import PythonAndReactAssistantAgent
from common.page import BasePage


class PythonAndReactAssistantPage(BasePage):
    agent = PythonAndReactAssistantAgent

    required_keys = ["OPENAI_API_KEY", "E2B_API_KEY"]

    def pre_render(self):
        if os.path.exists("application.flag"):
            os.remove("application.flag")

    def post_render(self):
        if os.path.exists("application.flag"):
            st.markdown(
                "<h3 class='fontStyle'>Application Preview</h3>", unsafe_allow_html=True
            )
            components.iframe(
                src=f"http://localhost:3000?t={int(time.time())}", height=500
            )
