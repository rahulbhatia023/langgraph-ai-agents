import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph_agents.research_analyst_agent import get_agent

st.title("Research Analyst AI Agent")

config = {"configurable": {"thread_id": "1"}}

agent = get_agent()

system_message = SystemMessage(
    content="""
    You are a research analyst AI agent.
    Always start the conversation with "Hi! I'm an AI research analyst helping you with your research."
    Then ask for the topic and the number of analysts to be involved in the research.
    After that, you can start the research process.
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def add_chat_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)


def stream_events(input):
    for event in agent.stream(
            input=input,
            config=config,
            stream_mode="updates"
    ):
        for k, v in event.items():
            if (k in ["agent", "finalize_report"]) and "messages" in v:
                m = v["messages"][-1]
                if (m.type == "ai" and not m.tool_calls) or m.type == "human":
                    add_chat_message(m.type, m.content)


if human_message := st.chat_input():
    add_chat_message("human", human_message)
    agent.update_state(config, {"messages": [HumanMessage(content=human_message)]}, as_node="agent")
    stream_events(None)
else:
    stream_events({"messages": [system_message]})
