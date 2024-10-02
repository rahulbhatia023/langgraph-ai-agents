import streamlit
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from tools.financial_advisor.last_quote import LastQuoteTool
from tools.financial_advisor.line_items import SearchLineItemsTool
from tools.financial_advisor.prices import PricesTool
from tools.financial_advisor.ticker_news import TickerNewsTool


def get_agent():
    polygon_api_key = streamlit.session_state["POLYGON_API_KEY"]
    financial_datasets_api_key = streamlit.session_state["FINANCIAL_DATASETS_API_KEY"]

    tools = [
        LastQuoteTool(polygon_api_key=polygon_api_key),
        PricesTool(financial_datasets_api_key=financial_datasets_api_key),
        TickerNewsTool(polygon_api_key=polygon_api_key),
        SearchLineItemsTool(financial_datasets_api_key=financial_datasets_api_key),
    ]

    llm = ChatOpenAI(
        model="gpt-4o", api_key=streamlit.session_state["OPENAI_API_KEY"], temperature=0
    ).bind_tools(tools=tools)

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
