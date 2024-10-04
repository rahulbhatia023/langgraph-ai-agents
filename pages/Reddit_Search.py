from agents.reddit_search_agent import get_agent
from common.pages_ui import render_agent_page, api_keys_missing

agent_name = "Reddit Search"

system_prompt = """
    You are a helpful assistant that helps users find information on Reddit.
    You can search for information on any topic and get relevant results.
    
    You need to use provided tools only to search for information on Reddit.
    
    Once you are provided with the information from the tool, summarize the contaxt and provide the best useful answer.
"""

if not api_keys_missing(
        [
            "OPENAI_API_KEY",
            "REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET",
            "REDDIT_USER_AGENT",
        ]
):
    render_agent_page(
        agent_name=agent_name, agent=get_agent(), system_prompt=system_prompt
    )
