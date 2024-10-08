from common.agent import BaseAgent


class DataQueryAssistantAgent(BaseAgent):
    name = "Data Query Assistant"

    system_prompt =  """
        You are a data analyst that can help summarize SQL tables and parse user questions about a database.
    """