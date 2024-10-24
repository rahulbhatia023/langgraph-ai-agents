from agents.podcast_script_writer_agent import PodcastScriptWriterAgent
from common.page import BasePage


class PodcastScriptWriterPage(BasePage):
    agent = PodcastScriptWriterAgent

    required_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"]


PodcastScriptWriterPage.display()
