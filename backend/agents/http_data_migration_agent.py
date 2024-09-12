import json
from typing import Annotated, TypedDict

import pandas as pd
import requests
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode


class HttpDataMigrationState(TypedDict):
    url: str
    data: str
    output_path: str
    messages: Annotated[list[str], add_messages]


def read_api(state):
    """The actual read api logic"""
    # https://api.ipify.org?format=json
    api_response = requests.get(state["url"]).json()
    return {"data": api_response}


def transform_data(state):
    """The actual transformation logic"""
    df = pd.DataFrame([state["data"]])
    df.rename(columns={"ip": "ip_address"}, inplace=True)
    transformed_api_response = df.to_json(orient="records")
    return {"data": transformed_api_response}


def write_data(state):
    """The actual write logic"""
    transformed_api_response = json.loads(state["data"])
    file_path = state["output_path"]
    with open(file_path, "w") as json_file:
        json.dump(transformed_api_response, json_file)


tools = [read_api, transform_data, write_data]
model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True).bind_tools(tools)


def call_model(state):
    system_message = SystemMessage(content="""
    You are an HTTP data migration agent. You need to perform below three steps:
    1. Read the data from the api. Ask for url from the user for this operation
    2. Transform the fetched data
    3. Write the transformed data. Ask for the output path from the user for this operation.
    """)
    response = model.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def ask_human(state):
    pass


def should_call_tools_or_human(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return "human"


def should_call_ai_or_quit(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content == "quit":
        return END
    else:
        return "ai"


graph = StateGraph(HttpDataMigrationState)

graph.add_node("ai", call_model)
graph.add_node("human", ask_human)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "ai")
graph.add_conditional_edges("ai", should_call_tools_or_human)
graph.add_conditional_edges("human", should_call_ai_or_quit)
graph.add_edge("tools", "ai")

agent = graph.compile(checkpointer=MemorySaver(), interrupt_before=["human"])


def get_agent():
    return agent
