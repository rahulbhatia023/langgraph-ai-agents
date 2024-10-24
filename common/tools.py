from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


def tavily_search(query, tavily_api_key, max_results=3):
    """Retrieve docs from web search"""
    return {
        "context": [
            "\n\n---\n\n".join(
                [
                    f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                    for doc in TavilySearchResults(
                        max_results=max_results,
                        api_wrapper=TavilySearchAPIWrapper(
                            tavily_api_key=tavily_api_key
                        ),
                    ).invoke(query)
                ]
            )
        ]
    }


def wikipedia_search(query, load_max_docs=2):
    """Retrieve docs from wikipedia"""
    return {
        "context": [
            "\n\n---\n\n".join(
                [
                    f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                    for doc in WikipediaLoader(
                        query=query,
                        load_max_docs=load_max_docs,
                    ).load()
                ]
            )
        ]
    }
