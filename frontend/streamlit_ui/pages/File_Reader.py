import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from langgraph_agents.file_reader_agent import agent

system_message = """
You are a helpful assistant that performs below steps:
1.  Ask user for the file path.
2.  Read the JSON file content.
3.  Once step 2 is done, provide a success message to the user.
4.  Ask user if they want to see the summary of the data.
5.  If the user says "yes", provide the total number of records in the file.
    Also show the schema for the data.

If you are not sure about anything apologise and gently say that you cannot do that.
"""


def run_agent(
    agent_name: str,
    agent: CompiledStateGraph,
    system_message: str,
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
        agent.update_state(
            config,
            {"messages": [HumanMessage(content=human_message)]},
            as_node=update_as_node,
        )
        stream_events(None)
    else:
        stream_events({"messages": [SystemMessage(content=system_message)]})


run_agent(
    agent_name="File Reader Agent",
    agent=agent,
    system_message=system_message,
    nodes_to_display=[],
    update_as_node="human",
)

with st.sidebar:
    st.image(agent.get_graph().draw_mermaid_png())
