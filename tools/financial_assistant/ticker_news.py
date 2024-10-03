from typing import Dict, Union, Annotated, Optional

import requests
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import BaseModel, Field, SkipValidation

POLYGON_BASE_URL = "https://api.polygon.io/"


class TickerNews(BaseModel):
    ticker: str = Field(..., description="The ticker of the stock.")
    limit: str = Field(default=3, description="Number of news to fetch")


class TickerNewsTool(BaseTool):
    polygon_api_key: str = Field(..., description="The Polygon API key.")
    name: str = "get-ticker-news"
    description: str = (
        "Get the most recent news articles relating to a stock ticker symbol, including a summary of the article and a link to the original source."
    )
    args_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = TickerNews
    return_direct: bool = True

    def _run(self, ticker: str, limit: str) -> Union[Dict, str]:
        url = f"{POLYGON_BASE_URL}v2/reference/news?ticker={ticker}&apiKey={self.polygon_api_key}&limit={limit}"
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status != "OK":
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)
