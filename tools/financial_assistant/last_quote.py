from typing import Dict, Union, Annotated, Optional

import requests
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import BaseModel, Field, SkipValidation

POLYGON_BASE_URL = "https://api.polygon.io/"


class LastQuote(BaseModel):
    ticker: str = Field(..., description="The ticker of the stock.")


class LastQuoteTool(BaseTool):
    polygon_api_key: str = Field(..., description="The Polygon API key.")
    name: str = "get-last-quote"
    description: str = (
        "Get the most recent National Best Bid and Offer (Quote) for a ticker."
    )
    args_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = LastQuote
    return_direct: bool = True

    def _run(self, ticker: str) -> Union[Dict, str]:
        url = f"{POLYGON_BASE_URL}v2/last/nbbo/{ticker}?apiKey={self.polygon_api_key}"
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status != "OK":
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)
