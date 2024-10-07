from agents.financial_assistant_agent import FinancialAssistantAgent
from common.page import BasePage


class FinancialAssistantPage(BasePage):
    agent = FinancialAssistantAgent

    required_keys = [
        "OPENAI_API_KEY",
        "FINANCIAL_DATASETS_API_KEY",
        "POLYGON_API_KEY",
        "TAVILY_API_KEY",
    ]


FinancialAssistantPage.display()
