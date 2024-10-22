import operator
from typing import TypedDict, Annotated, List

import streamlit as st
from langchain_core.messages import (
    SystemMessage,
    get_buffer_string,
    AIMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import Send
from pydantic import BaseModel, Field

from common.agent import BaseAgent
from common.tools import wikipedia_search, tavily_search
import google.generativeai as genai


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


class InterviewState(MessagesState):
    topic: str  # Topic of the podcast
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    section: str  # section transcript
    sections: list  # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class ResearchGraphState(MessagesState):
    topic: Annotated[str, operator.add]  # Research topic
    keywords: List  # Keywords
    max_analysts: int  # Number of analysts
    subtopics: List  # Subtopics
    sections: Annotated[list, operator.add]  # Send() API key
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report


question_instructions = """You are the host of a popular podcast and you are tasked with interviewing an expert to learn about a specific topic.

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.

2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {topic}

Begin by introducing the topic that fits your goals, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help"

Remember to stay in character throughout your response"""

search_instructions = """You will be given a conversation between a host of a popular podcast and an expert.

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query"""

answer_instructions = """You are an expert being interviewed by a popular podcast host.

Here is the analyst's focus area: {topic}.

Your goal is to answer a question posed by the interviewer.

To answer the question, use this context:
{context}

When answering questions, follow these steps

1. Use only the information provided in the context.

2. Do not introduce outside information or make assumptions beyond what is explicitly stated in the context.

3. The context includes sources on the topic of each document.

4. Make it interesting."""

section_writer_instructions = """You are an expert technical writer.

Your task is to create an interesting, easily digestible section of a podcast based on an interview.

1. Analyze the content of the interview

2. Create a script structure using markdown formatting

3. Make your title engaging based upon the focus area of the analyst:
{focus}

4. For the conversation:
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Mention turns of "Interviewer" and "Expert"
- Aim for approximately 1000 words maximum

5. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

report_writer_instructions = """You are a podcast script writer preparing a script for an episode on this overall topic:

{topic}

You have a dedicated researcher who has delved deep into various subtopics related to the main theme.
Your task:

1. You will be given a collection of part of script podcast from the researcher, each covering a different subtopic.
2. Carefully analyze the insights from each script.
3. Consolidate these into a crisp and engaging narrative that ties together the central ideas from all of the script, suitable for a podcast audience.
4. Weave the central points of each script into a cohesive and compelling story, ensuring a natural flow and smooth transitions between segments, creating a unified and insightful exploration of the overall topic.

To format your script:

1. Use markdown formatting.
2. Write in a conversational and engaging tone suitable for a podcast.
3. Seamlessly integrate the insights from each script into the narrative, using clear and concise language.
4. Use transitional phrases and signposting to guide the listener through the different subtopics.

Here are the scripts from the researcher to build your podcast script from:

{context}"""

intro_instructions = """You are a podcast producer crafting a captivating introduction for an upcoming episode on {topic}.
You will be given an outline of the episode's main segments.
Your job is to write a compelling and engaging introduction that hooks the listener and sets the stage for the discussion.
Include no unnecessary preamble or fluff.
Target around 300 words, using vivid language and intriguing questions to pique the listener's curiosity and preview the key themes and topics covered in the episode.
Use markdown formatting.
Create a catchy and relevant title for the episode and use the # header for the title.
Use ## Introduction as the section header for your introduction.
Here are the segments to draw upon for crafting your introduction: {formatted_str_sections}"""

conclusion_instructions = """You are a podcast producer crafting a memorable conclusion for an episode on {topic}.
You will be given an outline of the episode's main segments.
Your job is to write a concise and impactful conclusion that summarizes the key takeaways and leaves a lasting impression on the listener.
Include no unnecessary preamble or fluff.
Target around 200 words, highlighting the most important insights and offering a thought-provoking closing statement that encourages further reflection or action.
Use markdown formatting.
Use ## Conclusion as the section header for your conclusion.
Here are the segments to draw upon for crafting your conclusion: {formatted_str_sections}"""


class PodcastGeneratorAgent(BaseAgent):
    name = "Podcast Generator"

    interrupt_before = ["ask_topic"]

    update_as_node = "ask_topic"

    nodes_to_display = ["agent", "final_report"]

    @classmethod
    def update_graph_state(cls, human_message):
        return {"topic": human_message}

    @classmethod
    def get_graph(cls):
        def get_model(temp: float = 0.1, max_tokens: int = 100):
            return ChatOpenAI(
                model=cls.model,
                api_key=st.session_state["OPENAI_API_KEY"],
                temperature=temp,
                max_tokens=max_tokens,
            )

        def invoke_llm(state):
            response = get_model().invoke(state["messages"])
            return {"messages": [response]}

        def ask_topic(state):
            pass

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

        planning_builder = StateGraph(Planning)

        planning_builder.add_node("get_keywords", get_keywords)
        planning_builder.add_node("get_structure", get_structure)

        planning_builder.add_edge(START, "get_keywords")
        planning_builder.add_edge("get_keywords", "get_structure")
        planning_builder.add_edge("get_structure", END)

        podcast_gpt = get_model(max_tokens=1000)
        structured_llm = podcast_gpt.with_structured_output(SearchQuery)

        def generate_question(state: InterviewState):
            """Node to generate a question"""

            return {
                "messages": [
                    podcast_gpt.invoke(
                        [
                            SystemMessage(
                                content=question_instructions.format(
                                    topic=state["topic"]
                                )
                            )
                        ]
                        + state["messages"]
                    )
                ]
            }

        def search_web(state: InterviewState):
            return {
                "context": [
                    tavily_search(
                        query=structured_llm.invoke(
                            [SystemMessage(content=search_instructions)]
                            + state["messages"]
                        ).search_query
                    )
                ]
            }

        def search_wikipedia(state: InterviewState):
            return {
                "context": [
                    wikipedia_search(
                        query=structured_llm.invoke(
                            [SystemMessage(content=search_instructions)]
                            + state["messages"]
                        ).search_query
                    )
                ]
            }

        def generate_answer(state: InterviewState):
            answer = podcast_gpt.invoke(
                [
                    SystemMessage(
                        content=answer_instructions.format(
                            topic=state["topic"], context=state["context"]
                        )
                    )
                ]
                + state["messages"]
            )
            answer.name = "expert"

            return {"messages": [answer]}

        def save_podcast(state: InterviewState):
            return {"section": get_buffer_string(state["messages"])}

        def route_messages(state: InterviewState, name: str = "expert"):
            messages = state["messages"]
            max_num_turns = state.get("max_num_turns", 2)

            num_responses = len(
                [m for m in messages if isinstance(m, AIMessage) and m.name == name]
            )

            if num_responses >= max_num_turns:
                return "Save podcast"

            last_question = messages[-2]

            if "Thank you so much for your help" in last_question.content:
                return "Save podcast"
            return "Host question"

        generation_config = {
            "temperature": 0.21,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 5000,
            "response_mime_type": "text/plain",
        }

        genai.configure(api_key=st.session_state["GOOGLE_API_KEY"])

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        podcast_model = model.start_chat()

        def write_section(state: InterviewState):
            return {
                "sections": [
                    podcast_model.send_message(
                        section_writer_instructions.format(focus=state["topic"])
                        + f"Use this source to write your section: {state["section"]}"
                    ).text
                ]
            }

        interview_builder = StateGraph(InterviewState)

        interview_builder.add_node("Host question", generate_question)
        interview_builder.add_node("Web research", search_web)
        interview_builder.add_node("Wiki research", search_wikipedia)
        interview_builder.add_node("Expert answer", generate_answer)
        interview_builder.add_node("Save podcast", save_podcast)
        interview_builder.add_node("Write script", write_section)

        interview_builder.add_edge(START, "Host question")
        interview_builder.add_edge("Host question", "Web research")
        interview_builder.add_edge("Host question", "Wiki research")
        interview_builder.add_edge("Web research", "Expert answer")
        interview_builder.add_edge("Wiki research", "Expert answer")
        interview_builder.add_conditional_edges(
            "Expert answer", route_messages, ["Host question", "Save podcast"]
        )
        interview_builder.add_edge("Save podcast", "Write script")
        interview_builder.add_edge("Write script", END)

        def initiate_all_interviews(state: ResearchGraphState):
            return [
                Send(
                    "Create podcast",
                    {
                        "topic": state["topic"],
                        "messages": [
                            HumanMessage(
                                content=f"So you said you were researching about {subtopic}?"
                            )
                        ],
                    },
                )
                for subtopic in state["subtopics"]
            ]

        def write_report(state: ResearchGraphState):
            return {
                "content": podcast_model.send_message(
                    report_writer_instructions.format(
                        topic=state["topic"],
                        context="\n\n".join(
                            [f"{section}" for section in state["sections"]]
                        ),
                    )
                ).text
            }

        def write_introduction(state: ResearchGraphState):
            return {
                "introduction": podcast_model.send_message(
                    intro_instructions.format(
                        topic=state["topic"],
                        formatted_str_sections="\n\n".join(
                            [f"{section}" for section in state["sections"]]
                        ),
                    )
                ).text
            }

        def write_conclusion(state: ResearchGraphState):
            return {
                "conclusion": podcast_model.send_message(
                    conclusion_instructions.format(
                        topic=state["topic"],
                        formatted_str_sections="\n\n".join(
                            [f"{section}" for section in state["sections"]]
                        ),
                    )
                ).text
            }

        def finalize_report(state: ResearchGraphState):
            final_report = (
                state["introduction"]
                + "\n\n---\n\n"
                + state["content"]
                + "\n\n---\n\n"
                + state["conclusion"]
            )
            return {
                "final_report": final_report,
                "messages": [AIMessage(content=final_report)],
            }

        def Start_parallel(state):
            """No-op node that should be interrupted on"""
            pass

        builder = StateGraph(ResearchGraphState)

        builder.add_node("agent", invoke_llm)
        builder.add_node("ask_topic", ask_topic)
        builder.add_node("Planning", planning_builder.compile())
        builder.add_node("Start research", Start_parallel)
        builder.add_node("Create podcast", interview_builder.compile())
        builder.add_node("Write report", write_report)
        builder.add_node("Write introduction", write_introduction)
        builder.add_node("Write conclusion", write_conclusion)
        builder.add_node("Finalize podcast", finalize_report)

        builder.add_edge(START, "agent")
        builder.add_edge("agent", "ask_topic")
        builder.add_edge("ask_topic", "Planning")
        builder.add_edge("Planning", "Start research")
        builder.add_conditional_edges(
            "Start research", initiate_all_interviews, ["Planning", "Create podcast"]
        )
        builder.add_edge("Create podcast", "Write report")
        builder.add_edge("Create podcast", "Write introduction")
        builder.add_edge("Create podcast", "Write conclusion")
        builder.add_edge(
            ["Write introduction", "Write report", "Write conclusion"],
            "Finalize podcast",
        )
        builder.add_edge("Finalize podcast", END)

        return builder.compile(
            interrupt_before=cls.interrupt_before, checkpointer=MemorySaver()
        )
