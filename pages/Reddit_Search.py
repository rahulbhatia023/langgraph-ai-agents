from agents.reddit_search_agent import RedditSearchAgent
from common.page import BasePage


class RedditSearchPage(BasePage):
    agent = RedditSearchAgent

    required_keys = [
        "OPENAI_API_KEY",
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
    ]
