import streamlit
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from tools.reddit_search import RedditSearchTool


def get_agent():
    openai_api_key = streamlit.session_state["OPENAI_API_KEY"]
    reddit_client_id = streamlit.session_state["REDDIT_CLIENT_ID"]
    reddit_client_secret = streamlit.session_state["REDDIT_CLIENT_SECRET"]
    reddit_user_agent = streamlit.session_state["REDDIT_USER_AGENT"]

    tools = [
        RedditSearchTool(
            reddit_client_id=reddit_client_id, reddit_client_secret=reddit_client_secret, reddit_user_agent=reddit_user_agent
        )
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
