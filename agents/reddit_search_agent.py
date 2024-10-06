import streamlit

from common.agent import BaseAgent
from tools.reddit_search import RedditSearchTool


class RedditSearchAgent(BaseAgent):
    name = "Reddit Search"

    tools = [
        RedditSearchTool(
            reddit_client_id=streamlit.session_state["REDDIT_CLIENT_ID"],
            reddit_client_secret=streamlit.session_state["REDDIT_CLIENT_SECRET"],
            reddit_user_agent=streamlit.session_state["REDDIT_USER_AGENT"],
        )
    ]
