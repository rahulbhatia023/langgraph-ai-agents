from typing import Sequence

import streamlit
from langchain.prompts import Prompt
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from common.agent import BaseAgent
from tools.graph_rag import DocumentsRetrieverTool


class GraphRAGAgent(BaseAgent):
    name = "Graph RAG Agent"

    @classmethod
    def get_tools(cls) -> Sequence[BaseTool]:
        if (
            "uploaded_file" in streamlit.session_state
            and cls.name in streamlit.session_state["uploaded_file"]
            and streamlit.session_state["uploaded_file"][cls.name]
        ):
            return [
                DocumentsRetrieverTool(
                    pdf_file=streamlit.session_state["uploaded_file"][cls.name],
                    openai_api_key=streamlit.session_state["OPENAI_API_KEY"],
                    neo4j_uri=streamlit.session_state["NEO4J_URI"],
                    neo4j_username=streamlit.session_state["NEO4J_USERNAME"],
                    neo4j_password=streamlit.session_state["NEO4J_PASSWORD"],
                )
            ]
        else:
            return []

    @classmethod
    def get_graph(cls):
        openai_api_key = streamlit.session_state["OPENAI_API_KEY"]
        tools = cls.get_tools()

        llm = ChatOpenAI(
            model=cls.model,
            api_key=openai_api_key,
            base_url=cls.base_url,
            temperature=0,
        )

        llm_with_tools = llm.bind_tools(tools)

        def agent(state):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        def generate(state):
            messages = state["messages"]
            question = messages[0].content
            docs = messages[-1].content

            prompt = Prompt.from_template(
                """
                You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

                Question: {question} 
                
                Context: {context} 
                
                Answer:
                """
            )

            rag_chain = prompt | llm

            response = rag_chain.invoke({"context": docs, "question": question})

            return {"messages": [response]}

        graph = StateGraph(MessagesState)

        graph.add_node("agent", agent)
        graph.add_node("retrieve", ToolNode(tools))
        graph.add_node("generate", generate)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", tools_condition, {"tools": "retrieve", END: END}
        )
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()
