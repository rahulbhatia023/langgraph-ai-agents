import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from langgraph_agents.research_analyst_agent import get_agent

system_message = """
    You are a research analyst AI agent.
    Always start the conversation with "Hi! I'm an AI research analyst helping you with your research."
    Then ask for the topic and the number of analysts to be involved in the research.
    After that, you can start the research process.
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
    agent_name="Research Analyst AI Agent",
    agent=get_agent(),
    system_message=system_message,
    nodes_to_display=["agent", "finalize_report"],
    update_as_node="agent"
)
