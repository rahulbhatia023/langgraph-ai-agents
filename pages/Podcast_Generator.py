from agents.podcast_generator_agent import PodcastGeneratorAgent
from common.page import BasePage


class PodcastGeneratorPage(BasePage):
    agent = PodcastGeneratorAgent

    required_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY"]


PodcastGeneratorPage.display()
