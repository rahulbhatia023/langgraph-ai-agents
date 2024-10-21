from typing import TypedDict

import streamlit as st
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from pydantic import BaseModel, Field

from common.agent import BaseAgent


class Planning(TypedDict):
    topic: str
    keywords: list[str]
    subtopics: list[str]


class Keywords(BaseModel):
    """Answer with at least 5 keywords that you think are related to the topic"""

    keys: list = Field(description="list of at least 5 keywords related to the topic")


class Subtopics(BaseModel):
    """Answer with at least 5 subtopics related to the topic"""

    subtopics: list = Field(
        description="list of at least 5 subtopics related to the topic"
    )


class Structure(BaseModel):
    """Structure of the podcast having in account the topic and the keywords"""

    subtopics: list[Subtopics] = Field(
        description="5 subtopics that we will review in the podcast related to the Topic and the Keywords"
    )


class PodcastGeneratorAgent(BaseAgent):
    name = "Podcast Generator"

    @classmethod
    def get_graph(cls):
        def get_model(temp: float = 0.1, max_tokens: int = 100):
            return ChatOpenAI(
                model=cls.model,
                api_key=st.session_state["OPENAI_API_KEY"],
                temperature=temp,
                max_tokens=max_tokens,
            )

        def get_keywords(state: Planning):
            topic = state["topic"]
            messages = [
                SystemMessage(
                    content="Your task is to generate 5 relevant words about the following topic: "
                    + topic
                )
            ]
            message = (
                get_model(temp=0.1, max_tokens=50)
                .with_structured_output(Keywords)
                .invoke(messages)
            )
            return {"keywords": message.keys}

        def get_structure(state: Planning):
            topic = state["topic"]
            keywords = state["keywords"]
            messages = [
                SystemMessage(
                    content="You task is to generate 5 subtopics to make a podcast about the following topic: "
                    + topic
                    + "and the following keywords:"
                    + " ".join(keywords)
                )
            ]
            message = (
                get_model(temp=0.3, max_tokens=1000)
                .with_structured_output(Structure)
                .invoke(messages)
            )
            return {"subtopics": message.subtopics[0].subtopics}

        planning_graph = StateGraph(Planning)

        planning_graph.add_node("get_keywords", get_keywords)
        planning_graph.add_node("get_structure", get_structure)

        planning_graph.add_edge(START, "get_keywords")
        planning_graph.add_edge("get_keywords", "get_structure")
        planning_graph.add_edge("get_structure", END)

        return planning_graph.compile()
