import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from langgraph_agents.reddit_search_agent import agent


def run_agent(
    agent_name: str,
    agent: CompiledStateGraph,
    nodes_to_display: list[str],
    update_as_node: str,
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
            if m["role"] == "assistant" or m["role"] == "human":
                add_chat_message(m["role"], m["content"])

    def add_chat_message(role, content):
        st.session_state.messages.append({"role": role, "content": content})
        with st.chat_message(role):
            st.markdown(content)

    def stream_events(agent_input):
        if not nodes_to_display:
            for event in agent.stream(
                input=agent_input, config=config, stream_mode="updates"
            ):
                for k, v in event.items():
                    display_message(v)
        else:
            for event in agent.stream(
                input=agent_input, config=config, stream_mode="updates"
            ):
                for k, v in event.items():
                    if k in nodes_to_display:
                        display_message(v)

    if human_message := st.chat_input():
        add_chat_message("human", human_message)
        agent.update_state(
            config,
            {"input": human_message, "chat_history": []},
            as_node=update_as_node,
        )
        stream_events(None)


run_agent(
    agent_name="Reddit Search Agent",
    agent=agent,
    nodes_to_display=["user_input", "search", "final_answer"],
    update_as_node="user_input",
)

with st.sidebar:
    st.image(agent.get_graph().draw_mermaid_png())
