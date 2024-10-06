from typing import Sequence

import streamlit as st
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


class BaseAgent:
    name: str = None
    tools: Sequence[BaseTool] = None

    @classmethod
    def get_agent(cls):
        llm = ChatOpenAI(
            model="gpt-4o", api_key=st.session_state["OPENAI_API_KEY"], temperature=0
        )

        if cls.tools:
            llm = llm.bind_tools(tools=cls.tools)

        def call_llm(state):
            messages = state["messages"]
            response = llm.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(MessagesState)

        graph.add_node("agent", call_llm)
        graph.add_node("tools", ToolNode(cls.tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            source="agent",
            path=tools_condition,
            path_map={"tools": "tools", END: END},
        )
        graph.add_edge("tools", "agent")

        return graph.compile(checkpointer=MemorySaver())
