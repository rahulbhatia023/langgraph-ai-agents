import os
import time

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from e2b_code_interpreter import CodeInterpreter
from langchain_core.messages import AIMessage

from langgraph_agents.code_assistant_agent import agent

load_dotenv()

st.set_page_config(layout="wide")

st.session_state["messages"] = [{"role": "system", "content": """
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
        """}]
st.session_state["tool_text_list"] = []
st.session_state.chat_history = []

if os.path.exists("application.flag"):
    os.remove("application.flag")

sandbox = CodeInterpreter.create()
sandbox_id = sandbox.id
sandbox.keep_alive(300)

with open("sandbox_id.txt", "w") as f:
    f.write(sandbox_id)

col1, col2, col3, col4 = st.columns([0.05, 0.45, 0.05, 0.45])

with st.sidebar:
    st.subheader("This is the LangGraph workflow visualization of this application rendered in real-time.")
    st.image(agent.get_graph().draw_mermaid_png())

with col2:
    st.header('Chat Messages')
    messages = st.container(height=600, border=False)

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            messages.chat_message("user").write(message["content"]["text"])
        elif message["role"] == "assistant":
            if isinstance(message["content"], list):
                for part in message["content"]:
                    if part["type"] == "text":
                        messages.chat_message("assistant").markdown(part["text"])
                    elif part["type"] == "code":
                        messages.chat_message("assistant").code(part["code"])
            else:
                messages.chat_message("assistant").markdown(message["content"])

    user_prompt = st.chat_input()

    if user_prompt:
        messages.chat_message("user").write(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.chat_history.append({"role": "user", "content": {"type": "text", "text": user_prompt}})

        thread = {"configurable": {"thread_id": "1"}}
        ai_messages = ""
        for event in agent.stream(input=st.session_state.messages, config=thread, stream_mode="values"):
            for message in reversed(event):
                if not isinstance(message, AIMessage):
                    break
                else:
                    if (message.tool_calls and isinstance(message.content, list)) or (
                            message.tool_calls and isinstance(message.content, str)):
                        if isinstance(message.content, list):
                            for part in message.content:
                                if 'text' in part:
                                    ai_messages += str(part['text']) + "\n"
                                    st.session_state.tool_text_list.append({"type": "text", "text": part['text']})
                                    messages.chat_message("assistant").markdown(part['text'])
                        for tool_call in message.tool_calls:
                            if "code" in tool_call["args"]:
                                code_text = tool_call["args"]["code"]
                                ai_messages += code_text
                                st.session_state.tool_text_list.append({"type": "code", "code": code_text})
                                messages.chat_message("assistant").code(code_text)
                    else:
                        if os.path.exists("chart.png"):
                            col4.header('Images')
                            col4.image("chart.png")
                        ai_messages += str(message.content)
                        st.session_state.tool_text_list.append({"type": "text", "text": message.content})
                        messages.chat_message("assistant").markdown(message.content)
                        break
        st.session_state.messages.append({"role": "assistant", "content": ai_messages})
        st.session_state.chat_history.append({"role": "assistant", "content": st.session_state.tool_text_list})

if os.path.exists("application.flag"):
    with col4:
        st.header('Application Preview')
        react_app_url = f"http://localhost:3000?t={int(time.time())}"
        components.iframe(src=react_app_url, height=700)
