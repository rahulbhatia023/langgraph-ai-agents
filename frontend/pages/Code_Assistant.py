import os
import time

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from e2b_code_interpreter import CodeInterpreter
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph

from backend.agents.code_assistant_agent import agent

load_dotenv()

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
        .stApp {
            font-family: 'Poppins';
            background-color: #16423C;
        }
        
        .fontStyle {
            font-family: 'Poppins';
            color: #C4DAD2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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

if os.path.exists("e2b_sandbox.txt"):
    os.remove("e2b_sandbox.txt")
if os.path.exists("application.flag"):
    os.remove("application.flag")

sandbox = CodeInterpreter.create()
sandbox_id = sandbox.id
sandbox.keep_alive(300)

with open("e2b_sandbox.txt", "w") as f:
    f.write(sandbox_id)

with st.sidebar:
    st.markdown(
        "<h3 style='color:#E9EFEC;font-family: Poppins'>LangGraph Workflow Visualization</h3>",
        unsafe_allow_html=True,
    )

    st.markdown("""
        <style>
            [data-testid="stImage"] {
                border-radius: 10px;
                overflow: hidden;
            }
        </style>
        """, unsafe_allow_html=True)

    st.image(agent.get_graph().draw_mermaid_png(), use_column_width="always")


def run_agent(
        agent_name: str,
        agent: CompiledStateGraph,
        system_prompt: str,
        nodes_to_display: list[str],
):
    st.markdown(f"<h2 class='fontStyle'>{agent_name}</h2>", unsafe_allow_html=True)

    config = {"configurable": {"thread_id": "1"}}

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def display_message(v):
        if "messages" in v:
            m = v["messages"][-1]
            if (m.type == "ai" and not m.tool_calls) or m.type == "human":
                add_chat_message(m.type, m.content)

    def add_chat_message(role, content):
        st.session_state.messages.append({"role": role, "content": content})
        with st.chat_message(role):
            st.markdown(content)

    def stream_events(input):
        if not nodes_to_display:
            for event in agent.stream(
                    input=input, config=config, stream_mode="updates"
            ):
                for k, v in event.items():
                    display_message(v)

        else:
            for event in agent.stream(
                    input=input, config=config, stream_mode="updates"
            ):
                for k, v in event.items():
                    if k in nodes_to_display:
                        display_message(v)

    if human_message := st.chat_input():
        add_chat_message("human", human_message)
        stream_events(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_message),
                ]
            }
        )

    if os.path.exists("application.flag"):
        st.markdown("<h3 class='fontStyle'>Application Preview</h3>", unsafe_allow_html=True)
        components.iframe(src=f"http://localhost:3000?t={int(time.time())}", height=500)


run_agent(
    agent_name="Code Assistant",
    agent=agent,
    system_prompt=system_prompt,
    nodes_to_display=[],
)
