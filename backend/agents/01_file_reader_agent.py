"""
This code defines a LangChain agent that can read the content of a JSON file and provide a summary of the data.
The agent follows these steps:
1. Ask the user for the file path.
2. Read the JSON file content.
3. Ask the user if they want to see the summary of the data.
4. If the user says "yes", provide the total number of records in the file and show the schema for the data.
5. If the user says "no", end the conversation.
"""

from typing import TypedDict, Annotated

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode


class JSONFileReaderState(TypedDict):
    file_path: str
    file_content: str
    messages: Annotated[list[str], add_messages]


@tool
def read_file(file_path):
    """
    Read the content of a file and return it as a string.
    :param file_path: The path to the file to read.
    :return: The content of the file as a string.
    :rtype: str
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return {"file_content": content}


tools = [read_file]
model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True).bind_tools(tools)


def call_model(state):
    system_message = SystemMessage(content="""
    You are a helpful assistant that performs below steps:
    1.  Ask user for the file path.
    2.  Read the JSON file content.
    3.  Once step 2 is done, provide a success message to the user.
    4.  Ask user if they want to see the summary of the data.
    5.  If the user says "yes", provide the total number of records in the file.
        Also show the schema for the data.

    If you are not sure about anything apologise and gently say that you cannot do that.
    """)
    response = model.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def ask_human(state):
    pass


def ai_route(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return "human"


def human_route(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content == "quit":
        return END
    else:
        return "agent"


graph = StateGraph(JSONFileReaderState)

graph.add_node("agent", call_model)
graph.add_node("human", ask_human)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", ai_route)
graph.add_conditional_edges("human", human_route)
graph.add_edge("tools", "agent")

agent = graph.compile(checkpointer=MemorySaver(), interrupt_before=["human"])


def get_agent():
    return agent
