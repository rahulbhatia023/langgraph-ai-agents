from typing import Dict, Union, Annotated, Optional

import requests
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import BaseModel, Field, SkipValidation

FINANCIAL_DATASETS_BASE_URL = "https://api.financialdatasets.ai/"


class Prices(BaseModel):
    ticker: str = Field(..., description="The ticker of the stock.")
    start_date: str = Field(
        ...,
        description="The start of the price time window. Either a date with the format YYYY-MM-DD or a millisecond timestamp.",
    )
    end_date: str = Field(
        ...,
        description="The end of the aggregate time window. Either a date with the format YYYY-MM-DD or a millisecond timestamp.",
    )
    interval: str = Field(
        default="day",
        description="The time interval of the prices. Valid values are second', 'minute', 'day', 'week', 'month', 'quarter', 'year'.",
    )
    interval_multiplier: int = Field(
        default=1,
        description="The multiplier for the interval. For example, if interval is 'day' and interval_multiplier is 1, the prices will be daily. "
        "If interval is 'minute' and interval_multiplier is 5, the prices will be every 5 minutes.",
    )
    limit: int = Field(
        default=5000,
        description="The maximum number of prices to return. The default is 5000 and the maximum is 50000.",
    )


class PricesTool(BaseTool):
    financial_datasets_api_key: str = Field(
        ..., description="Financial Datasets API key."
    )
    name: str = "get-prices"
    description: str = "Get prices for a ticker over a given date range and interval."
    args_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = Prices
    return_direct: bool = True

    def _run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
        interval_multiplier: int = 1,
        limit: int = 5000,
    ) -> Union[Dict, str]:

        url = (
            f"{FINANCIAL_DATASETS_BASE_URL}prices"
            f"?ticker={ticker}"
            f"&start_date={start_date}"
            f"&end_date={end_date}"
            f"&interval={interval}"
            f"&interval_multiplier={interval_multiplier}"
            f"&limit={limit}"
        )

        try:
            response = requests.get(
                url, headers={"X-API-Key": self.financial_datasets_api_key}
            )
            data = response.json()
            return data
        except Exception as e:
            return {"ticker": ticker, "prices": [], "error": str(e)}
