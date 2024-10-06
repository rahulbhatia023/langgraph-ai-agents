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
