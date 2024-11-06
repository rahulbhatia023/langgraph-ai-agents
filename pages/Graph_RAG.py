from agents.graph_rag_agent import GraphRAGAgent
from common.page import BasePage


class GraphRAGPage(BasePage):
    agent = GraphRAGAgent

    required_keys = ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]

    show_file_uploader = True
    file_upload_label = "Upload PDF file"
    file_upload_type = ["pdf"]


GraphRAGPage.display()
