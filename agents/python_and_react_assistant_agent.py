import streamlit as st
from e2b_code_interpreter import CodeInterpreter

from common.agent import BaseAgent
from tools.python_and_react_assistant.execute_python import ExecutePythonTool
from tools.python_and_react_assistant.install_npm_dependencies import (
    install_npm_dependencies,
)
from tools.python_and_react_assistant.render_react import render_react
from tools.python_and_react_assistant.send_file_to_user import SendFileToUserTool


class PythonAndReactAssistantAgent(BaseAgent):
    name = "Python and React Assistant"

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

    tools = [
        ExecutePythonTool(
            e2b_sandbox_id=st.session_state.get("E2B_SANDBOX_ID", ""),
            e2b_api_key=st.session_state["E2B_API_KEY"],
        ),
        render_react,
        SendFileToUserTool(
            e2b_sandbox_id=st.session_state.get("E2B_SANDBOX_ID", ""),
            e2b_api_key=st.session_state["E2B_API_KEY"],
        ),
        install_npm_dependencies,
    ]
