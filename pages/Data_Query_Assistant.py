from agents.data_query_assistant_agent import DataQueryAssistantAgent
from common.page import BasePage


class DataQueryAssistantPage(BasePage):
    agent = DataQueryAssistantAgent

    required_keys = ["OPENAI_API_KEY"]


DataQueryAssistantPage.display()
