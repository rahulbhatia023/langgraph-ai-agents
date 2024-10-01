from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from tools.financial_advisor.last_quote import get_last_quote
from tools.financial_advisor.line_items import search_line_items
from tools.financial_advisor.prices import get_prices
from tools.financial_advisor.ticker_news import get_ticker_news

tools = [get_last_quote, get_prices, get_ticker_news, search_line_items]

llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools=tools)


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

agent = graph.compile(checkpointer=MemorySaver())
