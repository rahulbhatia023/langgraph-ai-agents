from datetime import datetime

import streamlit as st

from common.agent import BaseAgent
from tools.financial_assistant.last_quote import LastQuoteTool
from tools.financial_assistant.line_items import SearchLineItemsTool
from tools.financial_assistant.prices import PricesTool
from tools.financial_assistant.ticker_news import TickerNewsTool
from tools.financial_assistant.web_search import WebSearchTool


class FinancialAssistantAgent(BaseAgent):
    name = "Financial Assistant"

    system_prompt = f"""
        You are a highly capable financial assistant named FinanceGPT. Your purpose is to provide insightful and concise analysis to help users make informed financial decisions.

        Follow these steps:
        1. Identify the relevant financial data needed to answer the query.
        2. Use the available tools to retrieve the necessary data, such as stock financials, news, or aggregate data. 
        3. Use web-search tool in the last if the query cannot be answered because of authorization restrictions or any other reason while using other tools. Always consider today's date while using the web-search tool.
        4. Formulate a concise response that directly addresses the user's question, focusing on the most important findings from your analysis.

        Remember:
        - Today's date is {datetime.today().strftime("%d/%m/%Y")}.
        - Use general topic and include domains like "marketwatch.com" and "finance.yahoo.com" while using web-search tool.

        Your ultimate goal is to empower users with clear, actionable insights to navigate the financial landscape effectively.
    """

    tools = [
        LastQuoteTool(polygon_api_key=st.session_state["POLYGON_API_KEY"]),
        PricesTool(
            financial_datasets_api_key=st.session_state["FINANCIAL_DATASETS_API_KEY"]
        ),
        TickerNewsTool(polygon_api_key=st.session_state["POLYGON_API_KEY"]),
        SearchLineItemsTool(
            financial_datasets_api_key=st.session_state["FINANCIAL_DATASETS_API_KEY"]
        ),
        WebSearchTool(tavily_api_key=st.session_state["TAVILY_API_KEY"]),
    ]
