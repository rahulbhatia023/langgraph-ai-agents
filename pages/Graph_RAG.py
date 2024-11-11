import streamlit
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from agents.graph_rag_agent import GraphRAGAgent
from common.page import BasePage


class GraphRAGPage(BasePage):
    agent = GraphRAGAgent

    required_keys = {
        "OPENAI_API_KEY": "password",
        "NEO4J_URI": "default",
        "NEO4J_USERNAME": "default",
        "NEO4J_PASSWORD": "password",
    }

    show_file_uploader = True
    file_upload_label = "Upload PDF file"
    file_upload_type = ["pdf"]

    @classmethod
    def on_file_upload(cls, uploaded_file):
        graph_documents = LLMGraphTransformer(
            llm=ChatOpenAI(
                model="gpt-4o",
                api_key=streamlit.session_state["OPENAI_API_KEY"],
                temperature=0,
            )
        ).convert_to_graph_documents(documents=PyPDFLoader(uploaded_file).load())

        Neo4jGraph(
            url=streamlit.session_state["NEO4J_URI"],
            username=streamlit.session_state["NEO4J_USERNAME"],
            password=streamlit.session_state["NEO4J_PASSWORD"],
        ).add_graph_documents(
            graph_documents=graph_documents, baseEntityLabel=True, include_source=True
        )


GraphRAGPage.display()
