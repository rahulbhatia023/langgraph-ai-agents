from datetime import datetime

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from langgraph_agents.financial_advisor_agent import get_agent

system_message = f"""
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


def run_agent(
        agent_name: str,
        agent: CompiledStateGraph,
        system_message: str,
        nodes_to_display: list[str],
        update_as_node: str
):
    st.title(agent_name)

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
                    input=input,
                    config=config,
                    stream_mode="updates"
            ):
                for k, v in event.items():
                    display_message(v)
        else:
            for event in agent.stream(
                    input=input,
                    config=config,
                    stream_mode="updates"
            ):
                for k, v in event.items():
                    if k in nodes_to_display:
                        display_message(v)

    if human_message := st.chat_input():
        add_chat_message("human", human_message)
        agent.update_state(config, {"messages": [HumanMessage(content=human_message)]}, as_node=update_as_node)
        stream_events(None)
    else:
        stream_events({"messages": [SystemMessage(content=system_message)]})


run_agent(
    agent_name="Financial Advisor Agent",
    agent=get_agent(),
    system_message=system_message,
    nodes_to_display=[],
    update_as_node="human"
)
