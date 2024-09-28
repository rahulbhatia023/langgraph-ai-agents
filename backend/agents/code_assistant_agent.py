from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import MessageGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.tools.code_assistant.execute_python import execute_python
from backend.tools.code_assistant.install_npm_dependencies import (
    install_npm_dependencies,
)
from backend.tools.code_assistant.render_react import render_react
from backend.tools.code_assistant.send_file_to_user import send_file_to_user

load_dotenv()

tools = [execute_python, render_react, install_npm_dependencies, send_file_to_user]

llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4096)

llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto")

graph = MessageGraph()
graph.add_node("chatbot", llm_with_tools)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("chatbot")
graph.add_conditional_edges(
    source="chatbot",
    path=tools_condition,
    path_map={"tools": "tools", END: END},
)
graph.add_edge("tools", "chatbot")

agent = graph.compile(checkpointer=MemorySaver())
