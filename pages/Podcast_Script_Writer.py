from agents.podcast_script_writer_agent import PodcastScriptWriterAgent
from common.page import BasePage


class PodcastScriptWriterPage(BasePage):
    agent = PodcastScriptWriterAgent

    required_keys = {
        "OPENAI_API_KEY": "password",
        "GOOGLE_API_KEY": "password",
        "TAVILY_API_KEY": "password",
    }


PodcastScriptWriterPage.display()
