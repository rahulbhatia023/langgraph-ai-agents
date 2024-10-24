import operator
from typing import Annotated, List

import google.generativeai as genai
import streamlit
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

from common.agent import BaseAgent
from common.tools import wikipedia_search, tavily_search


class Planning(MessagesState):
    topic: str
    keywords: list[str]
    subtopics: list[str]


class InterviewState(MessagesState):
    topic: str  # Topic of the podcast
    subtopic: str
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    section: str  # section transcript
    sections: list  # Final key we duplicate in outer state for Send() API


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

Here is your topic of focus and set of goals: {subtopic} in regards with: {topic}

Begin by introducing the topic that fits your goals, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help"

Remember to stay in character throughout your response"""

search_instructions = """You will be given a conversation between a host of a popular podcast and an expert.

First, analyze the full conversation.

Pay particular attention to the final question posed by the host.

Convert this final question into an independent query to be used later for the web search. 

Do not include any preambles or the conversation in the generated query."""

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


class PodcastScriptWriterAgent(BaseAgent):
    name = "Podcast Script Writer"

    system_prompt = """You are an intelligent AI agent specialised in writing the script for a podcast for a specific topic.
    Just greet the user with a Hi and ask for the topic and then proceed accordingly. Please do not include any further details and preamble."""

    interrupt_before = ["ask_topic"]

    update_as_node = "ask_topic"

    nodes_to_display = ["agent", "Finalize podcast"]

    @classmethod
    def update_graph_state(cls, human_message):
        return {"topic": human_message}

    @classmethod
    def get_graph(cls):
        openai_api_key = streamlit.session_state["OPENAI_API_KEY"]
        tavily_api_key = streamlit.session_state["TAVILY_API_KEY"]

        def get_model(temp: float = 0.1, max_tokens: int = 100):
            return ChatOpenAI(
                model=cls.model,
                api_key=openai_api_key,
                temperature=temp,
                max_tokens=max_tokens,
                base_url=cls.base_url,
            )

        def invoke_llm(state):
            response = get_model().invoke(state["messages"])
            return {"messages": [response]}

        def ask_topic(state):
            pass

        def get_keywords(state: Planning):
            return {
                "keywords": get_model()
                .invoke(
                    [
                        SystemMessage(
                            content=f"Your task is to generate 5 comma separated relevant words about the following topic: {state["topic"]}"
                        )
                    ]
                    + state["messages"]
                )
                .content.split(",")
            }

        def get_structure(state: Planning):
            return {
                "subtopics": get_model()
                .invoke(
                    [
                        SystemMessage(
                            content=f"""You task is to generate 5 comma separated subtopics to make a podcast about the following topic: {state["topic"]}, and the following keywords: {",".join(state["keywords"])}.
                            Do not include any preamble in your response, just provide the comma separated subtopics."""
                        )
                    ]
                    + state["messages"]
                )
                .content.split(",")
            }

        planning_builder = StateGraph(Planning)

        planning_builder.add_node("get_keywords", get_keywords)
        planning_builder.add_node("get_structure", get_structure)

        planning_builder.add_edge(START, "get_keywords")
        planning_builder.add_edge("get_keywords", "get_structure")
        planning_builder.add_edge("get_structure", END)

        def generate_question(state: InterviewState):
            """Node to generate a question"""
            return {
                "messages": [
                    get_model(max_tokens=1000).invoke(
                        [
                            SystemMessage(
                                content=question_instructions.format(
                                    topic=state["topic"], subtopic=state["subtopic"]
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
                        query=get_model(max_tokens=1000)
                        .invoke(
                            [SystemMessage(content=search_instructions)]
                            + [state["messages"][-1]]
                        )
                        .content,
                        tavily_api_key=tavily_api_key,
                    )
                ]
            }

        def search_wikipedia(state: InterviewState):
            return {
                "context": [
                    wikipedia_search(
                        query=get_model(max_tokens=1000)
                        .invoke(
                            [SystemMessage(content=search_instructions)]
                            + state["messages"]
                        )
                        .content
                    )
                ]
            }

        def generate_answer(state: InterviewState):
            answer = get_model(max_tokens=1000).invoke(
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
                        "subtopic": subtopic,
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

        builder = StateGraph(ResearchGraphState)

        builder.add_node("agent", invoke_llm)
        builder.add_node("ask_topic", ask_topic)
        builder.add_node("Planning", planning_builder.compile())
        builder.add_node("Create podcast", interview_builder.compile())
        builder.add_node("Write report", write_report)
        builder.add_node("Write introduction", write_introduction)
        builder.add_node("Write conclusion", write_conclusion)
        builder.add_node("Finalize podcast", finalize_report)

        builder.add_edge(START, "agent")
        builder.add_edge("agent", "ask_topic")
        builder.add_edge("ask_topic", "Planning")
        builder.add_conditional_edges(
            "Planning", initiate_all_interviews, path_map=["Create podcast"]
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
