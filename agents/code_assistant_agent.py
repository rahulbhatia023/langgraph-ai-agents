from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from tools.code_assistant.execute_python import execute_python
from tools.code_assistant.install_npm_dependencies import (
    install_npm_dependencies,
)
from tools.code_assistant.render_react import render_react
from tools.code_assistant.send_file_to_user import send_file_to_user


def get_agent(openai_api_key: str):
    tools = [execute_python, render_react, send_file_to_user, install_npm_dependencies]

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
