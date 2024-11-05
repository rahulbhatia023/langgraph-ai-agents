from agents.simple_rag_agent import SimpleRAGAgent
from common.page import BasePage


class SimpleRAGPage(BasePage):
    agent = SimpleRAGAgent

    required_keys = ["OPENAI_API_KEY"]

    show_file_uploader = True
    file_upload_label = "Upload PDF file"
    file_upload_type = ["pdf"]


SimpleRAGPage.display()
