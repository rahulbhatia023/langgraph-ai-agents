import os
from typing import Dict, Union

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

POLYGON_BASE_URL = "https://api.polygon.io/"


class GetTickerNews(BaseModel):
    ticker: str = Field(..., description="The ticker of the stock.")
    limit: str = Field(default=3, description="Number of news to fetch")


@tool("get-ticker-news", args_schema=GetTickerNews, return_direct=True)
def get_ticker_news(ticker: str, limit: str) -> Union[Dict, str]:
    """
    Get the most recent news articles relating to a stock ticker symbol,
    including a summary of the article and a link to the original source.
    """
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Missing POLYGON_API_KEY secret.")

    url = f"{POLYGON_BASE_URL}v2/reference/news?ticker={ticker}&apiKey={api_key}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    status = data.get("status", None)
    if status != "OK":
        raise ValueError(f"API Error: {data}")

    return data.get("results", None)
