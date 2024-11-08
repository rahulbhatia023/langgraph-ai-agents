from agents.reddit_search_agent import RedditSearchAgent
from common.page import BasePage


class RedditSearchPage(BasePage):
    agent = RedditSearchAgent

    required_keys = {
        "OPENAI_API_KEY": "password",
        "REDDIT_CLIENT_ID": "password",
        "REDDIT_CLIENT_SECRET": "password",
        "REDDIT_USER_AGENT": "default",
    }


RedditSearchPage.display()
