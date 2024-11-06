from typing import Sequence

import streamlit
from langchain.prompts import Prompt
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from common.agent import BaseAgent
from tools.simple_rag import DocumentsRetrieverTool


class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class SimpleRAGAgent(BaseAgent):
    name = "Simple RAG Agent"

    system_prompt = """
    You are a helpful assistant. Answer the user's questions based on the tools provided.
    """

    nodes_to_display = ["agent", "generate"]

    @classmethod
    def get_tools(cls) -> Sequence[BaseTool]:
        if (
            "uploaded_file" in streamlit.session_state
            and streamlit.session_state["uploaded_file"][cls.name]
        ):
            return [
                DocumentsRetrieverTool(
                    pdf_file=streamlit.session_state["uploaded_file"][cls.name],
                    openai_api_key=streamlit.session_state["OPENAI_API_KEY"],
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

        def rewrite(state):
            human_message = f"""
            Look at the input and try to reason about the underlying semantic intent / meaning.
            Here is the initial question:
            {state["messages"][0].content}
            
            Formulate an improved question."""

            return {
                "messages": [
                    llm.invoke(
                        [
                            HumanMessage(
                                content=human_message,
                            )
                        ]
                    )
                ]
            }

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

            rag_chain = prompt | llm_with_tools

            response = rag_chain.invoke({"context": docs, "question": question})

            return {"messages": [response]}

        def grade_documents(state):
            prompt = Prompt.from_template(
                """You are a grader assessing relevance of a retrieved document to a user question.
                
                Here is the retrieved document:
                {context}
                
                Here is the user question: 
                {question}
                
                If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
            )

            chain = prompt | llm.with_structured_output(Grade)

            messages = state["messages"]

            scored_result = chain.invoke(
                {"question": messages[0].content, "context": messages[-1].content}
            )

            score = scored_result.binary_score

            if score == "yes":
                return "generate"

            else:
                return "rewrite"

        graph = StateGraph(MessagesState)

        graph.add_node("agent", agent)
        graph.add_node("retrieve", ToolNode(tools))
        graph.add_node("rewrite", rewrite)
        graph.add_node("generate", generate)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", tools_condition, {"tools": "retrieve", END: END}
        )
        graph.add_conditional_edges(
            "retrieve", grade_documents, ["generate", "rewrite"]
        )
        graph.add_edge("generate", END)
        graph.add_edge("rewrite", "agent")

        return graph.compile()
