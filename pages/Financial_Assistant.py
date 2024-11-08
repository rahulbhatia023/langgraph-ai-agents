from agents.financial_assistant_agent import FinancialAssistantAgent
from common.page import BasePage


class FinancialAssistantPage(BasePage):
    agent = FinancialAssistantAgent

    required_keys = {
        "OPENAI_API_KEY": "password",
        "FINANCIAL_DATASETS_API_KEY": "password",
        "POLYGON_API_KEY": "password",
        "TAVILY_API_KEY": "password",
    }


FinancialAssistantPage.display()
