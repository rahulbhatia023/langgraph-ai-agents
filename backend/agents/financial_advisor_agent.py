from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tools.financial_advisor.last_quote import get_last_quote
from tools.financial_advisor.line_items import search_line_items
from tools.financial_advisor.prices import get_prices
from tools.financial_advisor.ticker_news import get_ticker_news

load_dotenv()

tools = [get_last_quote, get_prices, get_ticker_news, search_line_items]

model = ChatOpenAI(model="gpt-4", temperature=0).bind_tools(tools)


def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def ask_human(state):
    pass


def agent_route(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "human"


def human_route(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content == "quit":
        return "quit"
    else:
        return "agent"


graph = StateGraph(MessagesState)

graph.add_node("agent", call_model)
graph.add_node("human", ask_human)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    agent_route,
    {
        "tools": "tools",
        "human": "human",
    },
)
graph.add_conditional_edges(
    "human",
    human_route,
    {
        "agent": "agent",
        "quit": END,
    },
)
graph.add_edge("tools", "agent")

agent = graph.compile(checkpointer=MemorySaver(), interrupt_before=["human"])
