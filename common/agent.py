from typing import Sequence

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


class BaseAgent:
    name: str = None
    system_prompt: str = None
    interrupt_before: list[str] = []
    tools: Sequence[BaseTool] = []

    model = "llama3.1:8b"
    api_key = "ollama"
    base_url = "http://localhost:11434/v1"

    @classmethod
    def get_graph(cls):
        llm = ChatOpenAI(
            model=cls.model, api_key=cls.api_key, base_url=cls.base_url, temperature=0
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

        return graph.compile(
            interrupt_before=cls.interrupt_before, checkpointer=MemorySaver()
        )
