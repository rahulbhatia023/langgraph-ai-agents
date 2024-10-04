import os
import time

import streamlit as st
import streamlit.components.v1 as components

from agents.python_and_react_assistant_agent import get_agent
from common.pages_ui import render_agent_page

agent_name = "Python and React Assistant"

system_prompt = """
        You are a Python and React expert. You can create React applications and run Python code in a Jupyter notebook. Here are some guidelines for this environment:
        - The python code runs in jupyter notebook.
        - Display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
        - You have access to the internet and can make api requests.
        - You also have access to the filesystem and can read/write files.
        - You can install any pip package when you need. But the usual packages for data analysis are already preinstalled. Use the `!pip install -q package_name` command to install a package.
        - You can run any python code you want, everything is running in a secure sandbox environment.
        - NEVER execute provided tools when you are asked to explain your code.
        - NEVER use `execute_python` tool when you are asked to create a react application. Use `render_react` tool instead.
        - Prioritize to use tailwindcss for styling your react components.
        - Always display the code to the user while generating the final response
        """

if os.path.exists("application.flag"):
    os.remove("application.flag")

render_agent_page(
    agent_name=agent_name,
    agent=get_agent(),
    system_prompt=system_prompt,
    required_api_keys=["OPENAI_API_KEY"],
)

if os.path.exists("application.flag"):
    st.markdown(
        "<h3 class='fontStyle'>Application Preview</h3>", unsafe_allow_html=True
    )
    components.iframe(src=f"http://localhost:3000?t={int(time.time())}", height=500)
