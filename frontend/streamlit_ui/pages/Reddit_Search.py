import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from langgraph_agents.reddit_search_agent import get_agent

system_message = """
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

Your goal is to provide the user with the best possible restaurant
recommendation. Including key information about why they should consider
visiting or ordering from the restaurant, and how they can do so, ie by
providing restaurant address, phone number, website, etc.

Note, when using a tool, you provide the tool name and the arguments to use
in JSON format. For each call, you MUST ONLY use one tool AND the response
format must ALWAYS be in the pattern:

```json
{
"name": "<tool_name>",
    "parameters": {"<tool_input_key>": <tool_input_value>}
}
```

Remember, NEVER use the search tool more than 3x as that can trigger
the nuclear annihilation system.

After using the search tool you must summarize your findings with the
final_answer tool. Note, if the user asks a question or says something
unrelated to restaurants, you must use the final_answer tool directly.
"""


def run_agent(
        agent_name: str,
        agent: CompiledStateGraph,
        input: dict,
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
            if m['role'] == "assistant" or m['role'] == "human":
                add_chat_message(m['role'], m['content'])

    def add_chat_message(role, content):
        st.session_state.messages.append({"role": role, "content": content})
        with st.chat_message(role):
            st.markdown(content)

    def stream_events(agent_input):
        if not nodes_to_display:
            for event in agent.stream(
                    input=agent_input,
                    config=config,
                    stream_mode="updates"
            ):
                for k, v in event.items():
                    display_message(v)
        else:
            for event in agent.stream(
                    input=agent_input,
                    config=config,
                    stream_mode="updates"
            ):
                for k, v in event.items():
                    if k in nodes_to_display:
                        display_message(v)

    if human_message := st.chat_input():
        add_chat_message("human", human_message)
        agent.update_state(config, {"messages": [{"role": "human", "content": human_message}]}, as_node=update_as_node)
        stream_events(None)
    else:
        stream_events(input)


run_agent(
    agent_name="Reddit Search Agent",
    agent=get_agent(),
    input={"messages": [{"role": "system", "content": system_message}], "chat_history": []},
    nodes_to_display=["agent", "user_input", "final_answer"],
    update_as_node="agent"
)
