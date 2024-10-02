from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from agents.financial_advisor_agent import get_agent

agent_name = "Financial Advisor"

system_prompt = f"""
    You are a highly capable financial assistant named FinanceGPT. Your purpose is to provide insightful and concise analysis to help users make informed financial decisions.
    
    Follow these steps:
    0. Ask the query from the user
    1. Identify the relevant financial data needed to answer the query.
    2. Use the available tools to retrieve the necessary data, such as stock financials, news, or aggregate data.
    3. Analyze the retrieved data and any generated charts to extract key insights and trends.
    4. Formulate a concise response that directly addresses the user's question, focusing on the most important findings from your analysis.
    
    Remember:
    - Today's date is {datetime.today().strftime("%Y %m %d")}.
    - Avoid simply regurgitating the raw data from the tools. Instead, provide a thoughtful interpretation and summary.
    - If the query cannot be satisfactorily answered using the available tools, kindly inform the user and suggest alternative resources or information they may need.
    
    Your ultimate goal is to empower users with clear, actionable insights to navigate the financial landscape effectively.
    
    Remember your goal is to answer the users query and provide a clear, actionable answer.
"""

st.set_page_config(page_title=agent_name, page_icon="🤖", layout="wide")

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


def api_keys_missing(keys):
    if_missing = False
    for key in keys:
        if key not in st.session_state or not st.session_state[key]:
            st.error(f"Please enter {key} in the home page.", icon="🚨")
            if_missing = True
    return if_missing


if not api_keys_missing(
    [
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "FINANCIAL_DATASETS_API_KEY",
        "POLYGON_API_KEY",
    ]
):
    agent = get_agent()

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

    def display_message(v):
        if "messages" in v:
            m = v["messages"][-1]
            if (m.type == "ai" and not m.tool_calls) or m.type == "human":
                add_chat_message(m.type, m.content)

    def add_chat_message(role, content):
        st.session_state.page_messages[agent_name].append(
            {"role": role, "content": content}
        )
        with st.chat_message(role):
            st.markdown(content)

    config = {"configurable": {"thread_id": "1"}}

    def stream_events(input):
        for event in agent.stream(input=input, config=config, stream_mode="updates"):
            for k, v in event.items():
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
