from agents.research_analyst_agent import ResearchAnalystAgent
from common.page import BasePage


class ResearchAnalystPage(BasePage):
    agent = ResearchAnalystAgent

    required_keys = {"OPENAI_API_KEY": "password", "TAVILY_API_KEY": "password"}


ResearchAnalystPage.display()
