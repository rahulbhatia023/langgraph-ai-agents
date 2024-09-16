import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph_agents.file_reader_agent import get_agent

st.title("File Reader AI Agent")

system_message = SystemMessage(content="""
You are a helpful assistant that performs below steps:
1.  Ask user for the file path.
2.  Read the JSON file content.
3.  Once step 2 is done, provide a success message to the user.
4.  Ask user if they want to see the summary of the data.
5.  If the user says "yes", provide the total number of records in the file.
    Also show the schema for the data.

If you are not sure about anything apologise and gently say that you cannot do that.
""")

config = {"configurable": {"thread_id": "1"}}

agent = get_agent()

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
            m = v["messages"][-1]
            if (m.type == "ai" and not m.tool_calls) or m.type == "human":
                add_chat_message(m.type, m.content)


if human_message := st.chat_input():
    add_chat_message("human", human_message)
    agent.update_state(config, {"messages": [HumanMessage(content=human_message)]}, as_node="human")
    stream_events(None)
else:
    stream_events({"messages": [system_message]})
