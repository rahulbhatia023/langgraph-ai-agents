import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from streamlit.commands.page_config import Layout, PageIcon


def api_keys_missing(keys):
    if_missing = False
    for key in keys:
        if key not in st.session_state or not st.session_state[key]:
            st.error(f"Please enter {key} in the home page.", icon="🚨")
            if_missing = True
    return if_missing


def add_chat_message(agent_name, role, content):
    st.session_state.page_messages[agent_name].append(
        {"role": role, "content": content}
    )
    with st.chat_message(role):
        st.markdown(content)


def display_message(agent_name, v):
    if "messages" in v:
        m = v["messages"][-1]
        add_chat_message(agent_name=agent_name, role=m.type, content=m.content)


def stream_events(agent_name, agent, input):
    config = {"configurable": {"thread_id": "1"}}
    for event in agent.stream(input=input, config=config, stream_mode="updates"):
        for k, v in event.items():
            display_message(agent_name, v)


def render_agent_page(
    agent_name: str,
    agent: CompiledStateGraph,
    system_prompt: str,
    page_icon: PageIcon = "🤖",
    layout: Layout = "wide",
):
    st.set_page_config(page_title=agent_name, page_icon=page_icon, layout=layout)

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

    st.markdown(f"<h2 class='fontStyle'>{agent_name}</h2>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            "<h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>LangGraph Workflow Visualization</h3>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <style>
                [data-testid="stImage"] {
                    border-radius: 10px;
                    overflow: hidden;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.image(agent.get_graph().draw_mermaid_png(), use_column_width="always")

    if "page_messages" not in st.session_state:
        st.session_state.page_messages = {}

    if agent_name not in st.session_state.page_messages:
        st.session_state.page_messages[agent_name] = []

    for message in st.session_state.page_messages[agent_name]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if human_message := st.chat_input():
        add_chat_message(agent_name=agent_name, role="human", content=human_message)

        stream_events(
            agent_name=agent_name,
            agent=agent,
            input={
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_message),
                ]
            },
        )
