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

    if (
        "E2B_SANDBOX_ID" not in st.session_state
        or not st.session_state["E2B_SANDBOX_ID"]
    ):
        with CodeInterpreter(api_key=st.session_state["E2B_API_KEY"]) as sandbox:
            sandbox_id = sandbox.id
            sandbox.keep_alive(300)
            st.session_state["E2B_SANDBOX_ID"] = sandbox_id

    tools = [
        ExecutePythonTool(
            e2b_sandbox_id=st.session_state["E2B_SANDBOX_ID"],
            e2b_api_key=st.session_state["E2B_API_KEY"],
        ),
        render_react,
        SendFileToUserTool(
            e2b_sandbox_id=st.session_state["E2B_SANDBOX_ID"],
            e2b_api_key=st.session_state["E2B_API_KEY"],
        ),
        install_npm_dependencies,
    ]
