from typing import Dict, Union, Annotated
from typing import List, Optional

import requests
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import BaseModel, Field, SkipValidation

TAVILY_BASE_URL = "https://api.tavily.com"


class WebSearch(BaseModel):
    query: str = Field(
        ..., description="The search query you want to execute with Tavily."
    )
    search_depth: Optional[str] = Field(
        default="advanced",
        description="The depth of the search. It can be 'basic' or 'advanced'.",
    )
    topic: Optional[str] = Field(
        default="general",
        description="The category of the search. Currently supports 'general' and 'news'.",
    )
    days: Optional[int] = Field(
        default=3,
        description="The number of days back from the current date to include in the search results. Only available for 'news' topic.",
    )
    max_results: Optional[int] = Field(
        default=3, description="The maximum number of search results to return."
    )
    include_images: Optional[bool] = Field(
        default=False,
        description="Include a list of query-related images in the response.",
    )
    include_image_descriptions: Optional[bool] = Field(
        default=False,
        description="When include_images is True, adds descriptive text for each image.",
    )
    include_answer: Optional[bool] = Field(
        default=True, description="Include a short answer to original query."
    )
    include_raw_content: Optional[bool] = Field(
        default=False,
        description="Include the cleaned and parsed HTML content of each search result.",
    )
    include_domains: Optional[List[str]] = Field(
        default=[],
        description="A list of domains to specifically include in the search results.",
    )
    exclude_domains: Optional[List[str]] = Field(
        default=[],
        description="A list of domains to specifically exclude from the search results.",
    )


class WebSearchTool(BaseTool):
    tavily_api_key: str = Field(..., description="The Tavily API key.")

    name: str = "web-search"
    description: str = (
        "This tool accesses real-time web data, news, articles and should be used when up-to-date information from the internet is required."
    )
    args_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = WebSearch
    return_direct: bool = True

    def _run(
        self,
        query: str,
        search_depth: str = "advanced",
        topic: str = "general",
        days: int = 3,
        max_results: int = 3,
        include_images: bool = False,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_domains: list = None,
        exclude_domains: list = None,
    ) -> Union[Dict, str]:
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "days": days,
            "max_results": max_results,
            "include_images": include_images,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_domains": include_domains if include_domains is not None else [],
            "exclude_domains": exclude_domains if exclude_domains is not None else [],
        }

        try:
            response = requests.post(
                f"{TAVILY_BASE_URL}/search",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
