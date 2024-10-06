import streamlit as st

from common.agent import BaseAgent
from tools.financial_assistant.last_quote import LastQuoteTool
from tools.financial_assistant.line_items import SearchLineItemsTool
from tools.financial_assistant.prices import PricesTool
from tools.financial_assistant.ticker_news import TickerNewsTool
from tools.financial_assistant.web_search import WebSearchTool


class FinancialAssistantAgent(BaseAgent):
    name = "Financial Assistant"

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
