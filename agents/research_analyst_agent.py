"""
Research is often laborious work offloaded to analysts. AI has considerable potential to assist with this.
However, research demands customization: raw LLM outputs are often poorly suited for real-world decision-making workflows.
Customized, AI-based research and report generation workflows are a promising way to address this.

Our goal is to build a lightweight, multi-agent system around chat models that customizes the research process.

Source Selection:
    Users can choose any set of input sources for their research.

Planning:
    Users provide a topic, and the system generates a team of AI analysts, each focusing on one sub-topic.
    Human-in-the-loop will be used to refine these sub-topics before research begins.

LLM Utilization:
    Each analyst will conduct in-depth interviews with an expert AI using the selected sources.
    The interview will be a multi-turn conversation to extract detailed insights as shown in the STORM paper.
    These interviews will be captured using sub-graphs with their internal state.

Research Process:
    Experts will gather information to answer analyst questions in parallel.
    And all interviews will be conducted simultaneously through map-reduce.

Output Format:
    The gathered insights from each interview will be synthesized into a final report.
    We'll use customizable prompts for the report, allowing for a flexible output format.
"""

import operator
from typing import List, Annotated

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    get_buffer_string,
    AIMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END, Send
from langgraph.graph import StateGraph, MessagesState
from pydantic import BaseModel, Field

from common.agent import BaseAgent


class UserInput(BaseModel):
    topic: str = Field(description="Research topic for the analysts.")
    max_analysts: int = Field(description="Maximum number of analysts to create.")


class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(description="Name of the analyst.")
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


class GenerateAnalystsState(MessagesState):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    analysts: List[Analyst]  # Analyst asking questions


class InterviewState(MessagesState):
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    analyst: Analyst  # Analyst asking questions
    interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class ResearchGraphState(MessagesState):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    analysts: List[Analyst]  # Analyst asking questions
    sections: Annotated[list, operator.add]  # Send() API key
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report


analyst_instructions = """
    You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

    1. First, review the research topic:
    {topic}
    
    2. Determine the most interesting themes based upon documents and / or feedback above.
    
    3. Pick the top {max_analysts} themes.
    
    4. Assign one analyst to each theme.
"""

question_instructions = """
    You are an analyst tasked with interviewing an expert to learn about a specific topic. 
    
    Your goal is boil down to interesting and specific insights related to your topic.
    
    1. Interesting: Insights that people will find surprising or non-obvious.
    
    2. Specific: Insights that avoid generalities and include specific examples from the expert.
    
    Here is your topic of focus and set of goals: {goals}
    
    Begin by introducing yourself using a name that fits your persona, and then ask your question.
    
    Continue to ask questions to drill down and refine your understanding of the topic.
    
    When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"
    
    Remember to stay in character throughout your response, reflecting the persona and goals provided to you.
"""

search_instructions = """
    You will be given a conversation between an analyst and an expert. 
    
    Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
    
    First, analyze the full conversation.
    
    Pay particular attention to the final question posed by the analyst.
    
    Convert this final question into a well-structured web search query
"""

answer_instructions = """
    You are an expert being interviewed by an analyst.
    
    Here is analyst area of focus: {goals}. 
    
    You goal is to answer a question posed by the interviewer.
    
    To answer question, use this context:
    
    {context}
    
    When answering questions, follow these guidelines:
    
    1. Use only the information provided in the context. 
    
    2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.
    
    3. The context contain sources at the topic of each individual document.
    
    4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 
    
    5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
    
    6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
    
    [1] assistant/docs/llama3_1.pdf, page 7 
    
    And skip the addition of the brackets as well as the Document source preamble in your citation.
"""

section_writer_instructions = """
    You are an expert technical writer. 
    
    Your task is to create a short, easily digestible section of a report based on a set of source documents.
    
    1. Analyze the content of the source documents: 
    - The name of each source document is at the start of the document, with the <Document tag.
    
    2. Create a report structure using markdown formatting:
    - Use ## for the section title
    - Use ### for sub-section headers
    
    3. Write the report following this structure:
    a. Title (## header)
    b. Summary (### header)
    c. Sources (### header)
    
    4. Make your title engaging based upon the focus area of the analyst: 
    {focus}
    
    5. For the summary section:
    - Set up summary with general background / context related to the focus area of the analyst
    - Emphasize what is novel, interesting, or surprising about insights gathered from the interview
    - Create a numbered list of source documents, as you use them
    - Do not mention the names of interviewers or experts
    - Aim for approximately 400 words maximum
    - Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
    
    6. In the Sources section:
    - Include all sources used in your report
    - Provide full links to relevant websites or specific document paths
    - Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
    - It will look like:
    
    ### Sources
    [1] Link or Document name
    [2] Link or Document name
    
    7. Be sure to combine sources. For example this is not correct:
    
    [3] https://ai.meta.com/blog/meta-llama-3-1/
    [4] https://ai.meta.com/blog/meta-llama-3-1/
    
    There should be no redundant sources. It should simply be:
    
    [3] https://ai.meta.com/blog/meta-llama-3-1/
    
    8. Final review:
    - Ensure the report follows the required structure
    - Include no preamble before the title of the report
    - Check that all guidelines have been followed
"""

report_writer_instructions = """
    You are a technical writer creating a report on this overall topic: 
    
    {topic}
    
    You have a team of analysts. Each analyst has done two things: 
    
    1. They conducted an interview with an expert on a specific sub-topic.
    2. They write up their finding into a memo.
    
    Your task: 
    
    1. You will be given a collection of memos from your analysts.
    2. Think carefully about the insights from each memo.
    3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
    4. Summarize the central points in each memo into a cohesive single narrative.
    
    To format your report:
    
    1. Use markdown formatting. 
    2. Include no pre-amble for the report.
    3. Use no sub-heading. 
    4. Start your report with a single title header: ## Insights
    5. Do not mention any analyst names in your report.
    6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
    7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
    8. List your sources in order and do not repeat.
    
    [1] Source 1
    [2] Source 2
    
    Here are the memos from your analysts to build your report from: 
    
    {context}
"""

intro_conclusion_instructions = """
    You are a technical writer finishing a report on {topic}
    
    You will be given all of the sections of the report.
    
    You job is to write a crisp and compelling introduction or conclusion section.
    
    The user will instruct you whether to write the introduction or conclusion.
    
    Include no pre-amble for either section.
    
    Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.
    
    Use markdown formatting. 
    
    For your introduction, create a compelling title and use the # header for the title.
    
    For your introduction, use ## Introduction as the section header. 
    
    For your conclusion, use ## Conclusion as the section header.
    
    Here are the sections to reflect on for writing: {formatted_str_sections}
"""


class ResearchAnalystAgent(BaseAgent):
    name = "Research Analyst"

    system_prompt = """
        You are a research analyst AI agent.
        Ask for the topic and the number of analysts to be involved in the research.
        After that, you can start the research process.
    """

    interrupt_before = ["user_input"]

    update_as_node = "agent"

    nodes_to_display = ["agent", "create_analysts", "finalize_report"]

    @classmethod
    def get_graph(cls):
        llm = ChatOpenAI(
            model=cls.model, api_key=cls.api_key, base_url=cls.base_url, temperature=0
        )

        graph = StateGraph(ResearchGraphState)

        def call_model(state: GenerateAnalystsState):
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        def user_input(state: GenerateAnalystsState):
            last_message = state["messages"][-1]
            structured_llm = llm.with_structured_output(UserInput)

            prompt = """
            Below is the input from the user:
            {last_message}

            Fetch the topic and number of analysts from the user's input.
            """

            response = structured_llm.invoke(prompt.format(last_message=last_message))

            return {"topic": response.topic, "max_analysts": response.max_analysts}

        def create_analysts(state: GenerateAnalystsState):
            """Create analysts"""

            topic = state["topic"]
            max_analysts = state["max_analysts"]

            # Enforce structured output
            structured_llm = llm.with_structured_output(Perspectives)

            # System message
            system_message = analyst_instructions.format(
                topic=topic,
                max_analysts=max_analysts,
            )

            # Generate question
            analysts = structured_llm.invoke(
                [SystemMessage(content=system_message)]
                + [HumanMessage(content="Generate the set of analysts.")]
            )

            prompt_analysts_details = """
            Below are the details of the analysts:
            {analysts}

            Convert these details in a pretty manner to show to the user.

            Mention to the user that these are the analysts that are participating in the research.
            And then mention the details. Avoid the header.
            """

            analysts_details = llm.invoke(
                prompt_analysts_details.format(
                    analysts=",".join([a.persona for a in analysts.analysts])
                )
            )

            # Write the list of analysis to state
            return {"messages": [analysts_details], "analysts": analysts.analysts}

        def ask_question(state: InterviewState):
            """Node to generate a question"""

            # Get state
            analyst = state["analyst"]
            messages = state["messages"]

            # Generate question
            system_message = question_instructions.format(goals=analyst.persona)
            question = llm.invoke([SystemMessage(content=system_message)] + messages)

            # Write messages to state
            return {"messages": [question]}

        def search_web(state: InterviewState):
            """Retrieve docs from web search"""

            # Search query
            structured_llm = llm.with_structured_output(SearchQuery)
            search_query = structured_llm.invoke(
                [SystemMessage(content=search_instructions)] + state["messages"]
            )

            # Search
            tavily_search = TavilySearchResults(max_results=3)
            search_docs = tavily_search.invoke(search_query.search_query)

            # Format
            formatted_search_docs = "\n\n---\n\n".join(
                [
                    f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                    for doc in search_docs
                ]
            )

            return {"context": [formatted_search_docs]}

        def search_wikipedia(state: InterviewState):
            """Retrieve docs from wikipedia"""

            # Search query
            structured_llm = llm.with_structured_output(SearchQuery)
            search_query = structured_llm.invoke(
                [search_instructions] + state["messages"]
            )

            # Search
            search_docs = WikipediaLoader(
                query=search_query.search_query, load_max_docs=2
            ).load()

            # Format
            formatted_search_docs = "\n\n---\n\n".join(
                [
                    f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                    for doc in search_docs
                ]
            )

            return {"context": [formatted_search_docs]}

        def answer_question(state: InterviewState):
            """Node to answer a question"""

            # Get state
            analyst = state["analyst"]
            messages = state["messages"]
            context = state["context"]

            # Answer question
            system_message = answer_instructions.format(
                goals=analyst.persona, context=context
            )
            answer = llm.invoke([SystemMessage(content=system_message)] + messages)

            # Name the message as coming from the expert
            answer.name = "expert"

            # Append it to state
            return {"messages": [answer]}

        def write_section(state: InterviewState):
            """Node to answer a question"""

            # Get state
            context = state["context"]
            analyst = state["analyst"]

            # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
            system_message = section_writer_instructions.format(
                focus=analyst.description
            )
            section = llm.invoke(
                [SystemMessage(content=system_message)]
                + [
                    HumanMessage(
                        content=f"Use this source to write your section: {context}"
                    )
                ]
            )

            # Append it to state
            return {"sections": [section.content]}

        def save_interview(state: InterviewState):
            """Save interviews"""

            # Get messages
            messages = state["messages"]

            # Convert interview to a string
            interview = get_buffer_string(messages)

            # Save to interviews key
            return {"interview": interview}

        def route_messages(state: InterviewState, name: str = "expert"):
            """Route between question and answer"""

            # Get messages
            messages = state["messages"]
            max_num_turns = state.get("max_num_turns", 2)

            # Check the number of expert answers
            num_responses = len(
                [m for m in messages if isinstance(m, AIMessage) and m.name == name]
            )

            # End if expert has answered more than the max turns
            if num_responses >= max_num_turns:
                return "save_interview"

            # This router is run after each question - answer pair
            # Get the last question asked to check if it signals the end of discussion
            last_question = messages[-2]

            if "Thank you so much for your help" in last_question.content:
                return "save_interview"
            return "ask_question"

        interview_builder = StateGraph(InterviewState)

        interview_builder.add_node("ask_question", ask_question)
        interview_builder.add_node("search_web", search_web)
        interview_builder.add_node("search_wikipedia", search_wikipedia)
        interview_builder.add_node("answer_question", answer_question)
        interview_builder.add_node("save_interview", save_interview)
        interview_builder.add_node("write_section", write_section)

        # Flow
        interview_builder.add_edge(START, "ask_question")
        interview_builder.add_edge("ask_question", "search_web")
        interview_builder.add_edge("ask_question", "search_wikipedia")
        interview_builder.add_edge("search_web", "answer_question")
        interview_builder.add_edge("search_wikipedia", "answer_question")
        interview_builder.add_conditional_edges(
            "answer_question",
            route_messages,
            {"ask_question": "ask_question", "save_interview": "save_interview"},
        )
        interview_builder.add_edge("save_interview", "write_section")
        interview_builder.add_edge("write_section", END)

        def write_report(state: ResearchGraphState):
            # Full set of sections
            sections = state["sections"]
            topic = state["topic"]

            # Concat all sections together
            formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

            # Summarize the sections into a final report
            system_message = report_writer_instructions.format(
                topic=topic, context=formatted_str_sections
            )
            report = llm.invoke(
                [SystemMessage(content=system_message)]
                + [HumanMessage(content=f"Write a report based upon these memos.")]
            )
            return {"content": report.content}

        def write_introduction(state: ResearchGraphState):
            # Full set of sections
            sections = state["sections"]
            topic = state["topic"]

            # Concat all sections together
            formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

            # Summarize the sections into a final report

            instructions = intro_conclusion_instructions.format(
                topic=topic, formatted_str_sections=formatted_str_sections
            )
            intro = llm.invoke(
                [instructions]
                + [HumanMessage(content=f"Write the report introduction")]
            )
            return {"introduction": intro.content}

        def write_conclusion(state: ResearchGraphState):
            # Full set of sections
            sections = state["sections"]
            topic = state["topic"]

            # Concat all sections together
            formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

            # Summarize the sections into a final report

            instructions = intro_conclusion_instructions.format(
                topic=topic, formatted_str_sections=formatted_str_sections
            )
            conclusion = llm.invoke(
                [instructions] + [HumanMessage(content=f"Write the report conclusion")]
            )
            return {"conclusion": conclusion.content}

        def finalize_report(state: ResearchGraphState):
            """This is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion"""

            # Save full final report
            content = state["content"]
            if content.startswith("## Insights"):
                content = content.strip("## Insights")
            if "## Sources" in content:
                try:
                    content, sources = content.split("\n## Sources\n")
                except:
                    sources = None
            else:
                sources = None

            final_report = (
                state["introduction"]
                + "\n\n---\n\n"
                + content
                + "\n\n---\n\n"
                + state["conclusion"]
            )
            if sources is not None:
                final_report += "\n\n## Sources\n" + sources

            return {
                "messages": [AIMessage(content=final_report)],
                "final_report": final_report,
            }

        def initiate_all_interviews(state: ResearchGraphState):
            """This is the "map" step where we run each interview sub-graph using Send API"""

            topic = state["topic"]

            return [
                Send(
                    "conduct_interview",
                    {
                        "analyst": analyst,
                        "messages": [
                            HumanMessage(
                                content=f"So you said you were writing an article on {topic}?"
                            )
                        ],
                    },
                )
                for analyst in state["analysts"]
            ]

        graph.add_node("agent", call_model)
        graph.add_node("user_input", user_input)
        graph.add_node("create_analysts", create_analysts)
        graph.add_node("conduct_interview", interview_builder.compile())
        graph.add_node("write_report", write_report)
        graph.add_node("write_introduction", write_introduction)
        graph.add_node("write_conclusion", write_conclusion)
        graph.add_node("finalize_report", finalize_report)

        graph.add_edge(START, "agent")
        graph.add_edge("agent", "user_input")
        graph.add_edge("user_input", "create_analysts")
        graph.add_conditional_edges("create_analysts", initiate_all_interviews)
        graph.add_edge("conduct_interview", "write_report")
        graph.add_edge("conduct_interview", "write_introduction")
        graph.add_edge("conduct_interview", "write_conclusion")
        graph.add_edge(
            ["write_conclusion", "write_report", "write_introduction"],
            "finalize_report",
        )
        graph.add_edge("finalize_report", END)

        return graph.compile(
            interrupt_before=cls.interrupt_before, checkpointer=MemorySaver()
        )
