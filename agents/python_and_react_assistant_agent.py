import streamlit
from e2b_code_interpreter import CodeInterpreter
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from tools.python_and_react_assistant.execute_python import ExecutePythonTool
from tools.python_and_react_assistant.install_npm_dependencies import (
    install_npm_dependencies,
)
from tools.python_and_react_assistant.render_react import render_react
from tools.python_and_react_assistant.send_file_to_user import SendFileToUserTool


def get_agent():
    e2b_api_key = streamlit.session_state["E2B_API_KEY"]
    openai_api_key = streamlit.session_state["OPENAI_API_KEY"]

    if (
        "E2B_SANDBOX_ID" not in streamlit.session_state
        or not streamlit.session_state["E2B_SANDBOX_ID"]
    ):
        with CodeInterpreter(api_key=e2b_api_key) as sandbox:
            sandbox_id = sandbox.id
            sandbox.keep_alive(300)
            streamlit.session_state["E2B_SANDBOX_ID"] = sandbox_id

    e2b_sandbox_id = streamlit.session_state["E2B_SANDBOX_ID"]

    tools = [
        ExecutePythonTool(e2b_sandbox_id=e2b_sandbox_id, e2b_api_key=e2b_api_key),
        render_react,
        SendFileToUserTool(e2b_sandbox_id=e2b_sandbox_id, e2b_api_key=e2b_api_key),
        install_npm_dependencies,
    ]

    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0).bind_tools(
        tools=tools
    )

    def call_llm(state):
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_llm)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        source="agent",
        path=tools_condition,
        path_map={"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())
