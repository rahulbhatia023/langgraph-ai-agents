from agents.data_query_assistant_agent import DataQueryAssistantAgent
from common.page import BasePage


class DataQueryAssistantPage(BasePage):
    agent = DataQueryAssistantAgent

    required_keys = {"OPENAI_API_KEY": "password"}

    # File Uploader
    show_file_uploader = True
    file_upload_label = "Upload SQLite DB file"
    file_upload_type = "sqlite"


DataQueryAssistantPage.display()
