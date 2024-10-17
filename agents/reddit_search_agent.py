import streamlit

from common.agent import BaseAgent
from tools.reddit_search import RedditSearchTool


class RedditSearchAgent(BaseAgent):
    name = "Reddit Search"

    system_prompt = """
        You are a helpful assistant that helps users find information on Reddit.
        You can search for information on any topic and get relevant results.

        You need to use provided tools only to search for information on Reddit.

        Once you are provided with the information from the tool, summarize the contaxt and provide the best useful answer.
    """

    @classmethod
    def get_tools(cls):
        return [
            RedditSearchTool(
                reddit_client_id=streamlit.session_state["REDDIT_CLIENT_ID"],
                reddit_client_secret=streamlit.session_state["REDDIT_CLIENT_SECRET"],
                reddit_user_agent=streamlit.session_state["REDDIT_USER_AGENT"],
            )
        ]
