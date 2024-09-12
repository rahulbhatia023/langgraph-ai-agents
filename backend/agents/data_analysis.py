import json
import os
import re
import subprocess
from pathlib import Path
from typing import Annotated, List, Optional, Dict, TypedDict, Sequence, Literal

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.tools import tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
from openai import InternalServerError
from pydantic.v1 import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Load environment variables
load_dotenv()

# Set up API keys and environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
WORKING_DIRECTORY = os.getenv("WORKING_DIRECTORY", "./data_analysis/")

# Validate critical environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
if not LANGCHAIN_API_KEY:
    raise ValueError("LANGCHAIN_API_KEY is not set in the environment variables.")

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-Agent Data Analysis System"

# Initialize language models
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4096)
    power_llm = ChatOpenAI(model="gpt-4o", temperature=0.5, max_tokens=4096)
    json_llm = ChatOpenAI(
        model="gpt-4o",
        model_kwargs={"response_format": {"type": "json_object"}},
        temperature=0,
        max_tokens=4096,
    )
    print("Language models initialized successfully.")
except Exception as e:
    print(f"Error initializing language models: {str(e)}")
    raise

# Ensure working directory exists
if not os.path.exists(WORKING_DIRECTORY):
    os.makedirs(WORKING_DIRECTORY)
    print(f"Created working directory: {WORKING_DIRECTORY}")

print("Initialization complete.")


class State(TypedDict):
    """Pydantic model for the entire state structure."""

    # The sequence of messages exchanged in the conversation
    messages: Sequence[BaseMessage]

    # The complete content of the research hypothesis
    hypothesis: str = ""

    # The complete content of the research process
    process: str = ""

    # next process
    process_decision: str = ""

    # The current state of data visualization planning and execution
    visualization_state: str = ""

    # The current state of the search process, including queries and results
    searcher_state: str = ""

    # The current state of Coder development, including scripts and outputs
    code_state: str = ""

    # The content of the report sections being written
    report_section: str = ""

    # The feedback and comments from the quality review process
    quality_review: str = ""

    # A boolean flag indicating if the current output requires revision
    needs_revision: bool = False

    # The identifier of the agent who sent the last message
    sender: str = ""


class NoteState(BaseModel):
    """Pydantic model for the entire state structure."""

    messages: Sequence[BaseMessage] = Field(
        default_factory=list, description="List of message dictionaries"
    )
    hypothesis: str = Field(default="", description="Current research hypothesis")
    process: str = Field(default="", description="Current research process")
    process_decision: str = Field(
        default="", description="Decision about the next process step"
    )
    visualization_state: str = Field(
        default="", description="Current state of data visualization"
    )
    searcher_state: str = Field(
        default="", description="Current state of the search process"
    )
    code_state: str = Field(default="", description="Current state of code development")
    report_section: str = Field(
        default="", description="Content of the report sections"
    )
    quality_review: str = Field(default="", description="Feedback from quality review")
    needs_revision: bool = Field(
        default=False, description="Flag indicating if revision is needed"
    )
    sender: str = Field(default="", description="Identifier of the last message sender")


@tool
def list_directory_contents(directory: str = "./data_analysis/") -> str:
    """
    List the contents of the specified directory.

    Args:
        directory (str): The path to the directory to list. Defaults to the data storage directory.

    Returns:
        str: A string representation of the directory contents.
    """
    try:
        print(f"Listing contents of directory: {directory}")
        contents = os.listdir(directory)
        print(f"Directory contents: {contents}")
        return f"Directory contents :\n" + "\n".join(contents)
    except Exception as e:
        print(f"Error listing directory contents: {str(e)}")
        return f"Error listing directory contents: {str(e)}"


@tool
def collect_data(data_path: Annotated[str, "Path to the CSV file"] = "./data.csv"):
    """
    Collect data from a CSV file.

    This function attempts to read a CSV file using different encodings.

    Returns:
    pandas.DataFrame: The data read from the CSV file.

    Raises:
    ValueError: If unable to read the file with any of the provided encodings.
    """
    if WORKING_DIRECTORY not in data_path:
        data_path = os.path.join(WORKING_DIRECTORY, data_path)
    else:
        data_path = data_path
    print(f"Attempting to read CSV file: {data_path}")
    data = pd.read_csv(data_path)
    print(f"Successfully read CSV file")
    return data


@tool
def google_search(query: Annotated[str, "The search query to use"]) -> str:
    """
    Perform a Google search based on the given query and return the top 5 results.

    This function uses Selenium to perform a headless Google search and BeautifulSoup to parse the results.

    Args:
    query (str): The search query to use.

    Returns:
    str: A string containing the titles, snippets, and links of the top 5 search results.

    Raises:
    Exception: If there's an error during the search process.
    """
    try:
        print(f"Performing Google search for query: {query}")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = Service(os.getenv("CHROMEDRIVER_PATH", "./chromedriver/chromedriver"))

        with webdriver.Chrome(options=chrome_options, service=service) as driver:
            url = f"https://www.google.com/search?q={query}"
            print(f"Accessing URL: {url}")
            driver.get(url)
            html = driver.page_source

        soup = BeautifulSoup(html, "html.parser")
        search_results = soup.select(".g")
        search = ""
        for result in search_results[:5]:
            title_element = result.select_one("h3")
            title = title_element.text if title_element else "No Title"
            snippet_element = result.select_one(".VwiC3b")
            snippet = snippet_element.text if snippet_element else "No Snippet"
            link_element = result.select_one("a")
            link = link_element["href"] if link_element else "No Link"
            search += f"{title}\n{snippet}\n{link}\n\n"

        print("Google search completed successfully")
        return search
    except Exception as e:
        print(f"Error during Google search: {str(e)}")
        return f"Error: {e}"


@tool
def FireCrawl_scrape_webpages(
    urls: Annotated[List[str], "List of URLs to scrape"]
) -> str:
    """
    Scrape the provided web pages for detailed information using FireCrawlLoader.

    This function uses the FireCrawlLoader to load and scrape the content of the provided URLs.

    Args:
    urls (List[str]): A list of URLs to scrape.

    Returns:
    Any: The result of the FireCrawlLoader's load operation.

    Raises:
    Exception: If there's an error during the scraping process.
    """
    try:
        print(f"Scraping webpages using FireCrawl: {urls}")
        loader = FireCrawlLoader(
            api_key=os.getenv("FIRECRAWL_API_KEY"), url=urls, mode="scrape"
        )
        result = loader.load()
        print("FireCrawl scraping completed successfully")
        return result
    except Exception as e:
        print(f"Error during FireCrawl scraping: {str(e)}")
        return f"Error: {e}"


@tool
def read_document(
    file_name: Annotated[str, "Name of the file to read"],
    start: Annotated[Optional[int], "Starting line number to read from"] = None,
    end: Annotated[Optional[int], "Ending line number to read to"] = None,
) -> str:
    """
    Read the specified document.

    This function reads a document from the specified file and returns its content.
    Optionally, it can return a specific range of lines.

    Returns:
    str: The content of the document or an error message.
    """
    try:
        if WORKING_DIRECTORY not in file_name:
            file_path = os.path.join(WORKING_DIRECTORY, file_name)
        else:
            file_path = file_name
        print(f"Reading document: {file_path}")
        with open(file_path, "r") as file:
            lines = file.readlines()
        if start is None:
            start = 0
        content = "\n".join(lines[start:end])
        print(f"Document read successfully: {file_path}")
        return content
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return f"Error: The file {file_name} was not found."
    except Exception as e:
        print(f"Error while reading document: {str(e)}")
        return f"Error while reading document: {str(e)}"


@tool
def execute_code(
    input_code: Annotated[str, "The Python code to execute."],
    codefile_name: Annotated[str, "The Python code file name or full path."],
):
    """
    Execute Python code and return the result.

    This function takes Python code as input, writes it to a file, executes it,
    and returns the output or any errors encountered during execution.

    Args:
    input_code (str): The Python code to be executed.
    codefile_name (str): The name of the file to save the code in, or the full path.

    Returns:
    dict: A dictionary containing the execution result, output, and file path.
    """
    try:
        # Get the absolute path of the storage directory
        storage_path = os.path.abspath(os.getenv("STORAGE_PATH", "./data_analysis"))

        # Check if codefile_name is already an absolute path
        if os.path.isabs(codefile_name):
            code_file_path = codefile_name
        else:
            # Ensure we're not adding 'data_analysis' multiple times
            if codefile_name.startswith("data_analysis"):
                code_file_path = os.path.join(
                    os.path.dirname(storage_path), codefile_name
                )
            else:
                code_file_path = os.path.join(storage_path, codefile_name)

        # Normalize the path
        code_file_path = os.path.normpath(code_file_path)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(code_file_path), exist_ok=True)

        # Write the input code to the file
        with open(code_file_path, "w") as code_file:
            code_file.write(input_code)

        print(f"Code written to file: {code_file_path}")

        # Construct the command to activate the virtual environment and run the script
        conda_path = os.getenv("CONDA_PATH", "/home/user/anaconda3")
        conda_env = os.getenv("CONDA_ENV", "base")

        # Construct the command to activate the Conda environment and execute the given command
        source = f"source {conda_path}/etc/profile.d/conda.sh"
        conda_activate = f"conda activate {conda_env}"
        python_cmd = f"python {code_file_path}"
        full_command = f"{source} && {conda_activate} && {python_cmd}"

        # Execute the code using subprocess for security
        result = subprocess.run(
            ["/bin/bash", "-c", full_command],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(
                code_file_path
            ),  # Set the working directory to the file's directory
        )

        # Capture standard output and error output
        output = result.stdout
        error_output = result.stderr

        if result.returncode == 0:
            print("Code executed successfully")
            print(output)
            return {
                "result": "Code executed successfully",
                "output": output
                + "\n\nIf you have completed all tasks, respond with FINAL ANSWER.",
                "file_path": code_file_path,
            }
        else:
            print(f"Code execution failed: {error_output}")
            print(error_output)
            return {
                "result": "Failed to execute",
                "error": error_output,
                "file_path": code_file_path,
            }
    except Exception as e:
        print("An error occurred while executing code")
        return {
            "result": "Error occurred",
            "error": str(e),
            "file_path": code_file_path if "code_file_path" in locals() else "Unknown",
        }


@tool
def execute_command(
    command: Annotated[str, "Command to be executed."]
) -> Annotated[str, "Output of the command."]:
    """
    Execute a command in a specified Conda environment and return its output.

    This function activates a Conda environment , executes the given command,
    and returns the output or any errors encountered during execution.
    Please use pip to install the package.
    Args:
    command (str): The command to be executed in the Conda environment.

    Returns:
    str: The output of the command or an error message.
    """
    try:
        # Get Conda-related paths from environment variables
        conda_path = os.getenv("CONDA_PATH", "/home/user/anaconda3")
        conda_env = os.getenv("CONDA_ENV", "base")

        # Construct the command to activate the Conda environment and execute the given command
        source = f"source {conda_path}/etc/profile.d/conda.sh"
        conda_activate = f"conda activate {conda_env}"
        full_command = f"{source} && {conda_activate} && {command}"

        print(f"Executing command: {command}")

        # Execute the command and capture the output
        result = subprocess.run(
            full_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable="/bin/bash",
        )
        print("Command executed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr}")
        return f"Error: {e.stderr}"


@tool
def create_document(
    points: Annotated[List[str], "List of points to be included in the document"],
    file_name: Annotated[str, "Name of the file to save the document"],
) -> str:
    """
    Create and save a text document in Markdown format.

    This function takes a list of points and writes them as numbered items in a Markdown file.

    Returns:
    str: A message indicating where the outline was saved or an error message.
    """
    try:
        if WORKING_DIRECTORY not in file_name:
            file_path = os.path.join(WORKING_DIRECTORY, file_name)
        else:
            file_path = file_name
        print(f"Creating document: {file_path}")
        with open(file_path, "w") as file:
            for i, point in enumerate(points):
                file.write(f"{i + 1}. {point}\n")
        print(f"Document created successfully: {file_path}")
        return f"Outline saved to {file_path}"
    except Exception as e:
        print(f"Error while saving outline: {str(e)}")
        return f"Error while saving outline: {str(e)}"


@tool
def edit_document(
    file_name: Annotated[str, "Name of the file to edit"],
    inserts: Annotated[Dict[int, str], "Dictionary of line numbers and text to insert"],
) -> str:
    """
    Edit a document by inserting text at specific line numbers.

    This function reads an existing document, inserts new text at specified line numbers,
    and saves the modified document.

    Args:
        file_name (str): Name of the file to edit.
        inserts (Dict[int, str]): Dictionary where keys are line numbers and values are text to insert.

    Returns:
        str: A message indicating the result of the operation.

    Example:
        file_name = "example.txt"
        inserts = {
            1: "This is the first line to insert.",
            3: "This is the third line to insert."
        }
        result = edit_document(file_name=file_name, inserts=inserts)
        print(result)
        # Output: "Document edited and saved to /path/to/example.txt"
    """
    try:
        if WORKING_DIRECTORY not in file_name:
            file_path = os.path.join(WORKING_DIRECTORY, file_name)
        else:
            file_path = file_name
        print(f"Editing document: {file_path}")
        with open(file_path, "r") as file:
            lines = file.readlines()

        sorted_inserts = sorted(inserts.items())

        for line_number, text in sorted_inserts:
            if 1 <= line_number <= len(lines) + 1:
                lines.insert(line_number - 1, text + "\n")
            else:
                print(f"Line number out of range: {line_number}")
                return f"Error: Line number {line_number} is out of range."

        with open(file_path, "w") as file:
            file.writelines(lines)

        print(f"Document edited successfully: {file_path}")
        return f"Document edited and saved to {file_path}"
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return f"Error: The file {file_name} was not found."
    except Exception as e:
        print(f"Error while editing document: {str(e)}")
        return f"Error while editing document: {str(e)}"


def create_agent(
    llm: ChatOpenAI,
    tools: list[tool],
    system_message: str,
    team_members: list[str],
    working_directory: str = "./data_analysis/",
) -> AgentExecutor:
    """
    Create an agent with the given language model, tools, system message, and team members.

    Parameters:
        llm (ChatOpenAI): The language model to use for the agent.
        tools (list[tool]): A list of tools the agent can use.
        system_message (str): A message defining the agent's role and tasks.
        team_members (list[str]): A list of team member roles for collaboration.
        working_directory (str): The directory where the agent's data will be stored.

    Returns:
        AgentExecutor: An executor that manages the agent's task execution.
    """

    print("Creating agent")

    # Ensure the ListDirectoryContents tool is available
    if list_directory_contents not in tools:
        tools.append(list_directory_contents)

    # Prepare the tool names and team members for the system prompt
    tool_names = ", ".join([tool.name for tool in tools])
    team_members_str = ", ".join(team_members)

    # List the initial contents of the working directory
    initial_directory_contents = list_directory_contents(working_directory)

    # Create the system prompt for the agent
    system_prompt = (
        "You are a specialized AI assistant in a data analysis team. "
        "Your role is to complete specific tasks in the research process. "
        "Use the provided tools to make progress on your task. "
        "If you can't fully complete a task, explain what you've done and what's needed next. "
        "Always aim for accurate and clear outputs. "
        f"You have access to the following tools: {tool_names}. "
        f"Your specific role: {system_message}\n"
        "Work autonomously according to your specialty, using the tools available to you. "
        "Do not ask for clarification. "
        "Your other team members (and other teams) will collaborate with you based on their specialties. "
        f"You are chosen for a reason! You are one of the following team members: {team_members_str}.\n"
        f"The initial contents of your working directory are:\n{initial_directory_contents}\n"
        "Use the ListDirectoryContents tool to check for updates in the directory contents when needed."
    )

    # Define the prompt structure with placeholders for dynamic content
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("ai", "hypothesis: {hypothesis}"),
            ("ai", "process: {process}"),
            ("ai", "process_decision: {process_decision}"),
            ("ai", "visualization_state: {visualization_state}"),
            ("ai", "searcher_state: {searcher_state}"),
            ("ai", "code_state: {code_state}"),
            ("ai", "report_section: {report_section}"),
            ("ai", "quality_review: {quality_review}"),
            ("ai", "needs_revision: {needs_revision}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent using the defined prompt and tools
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    print("Agent created successfully")

    # Return an executor to manage the agent's task execution
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)


def create_supervisor(
    llm: ChatOpenAI, system_prompt: str, members: list[str]
) -> AgentExecutor:
    print("Creating supervisor")
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    print("Supervisor created successfully")
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )


def create_note_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> AgentExecutor:
    """
    Create a Note Agent that updates the entire state.
    """
    print("Creating note agent")
    parser = PydanticOutputParser(pydantic_object=NoteState)
    output_format = parser.get_format_instructions()
    escaped_output_format = output_format.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
                + "\n\nPlease format your response as a JSON object with the following structure:\n"
                + escaped_output_format,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    print(f"Note agent prompt: {prompt}")
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    print("Note agent created successfully")
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=False,
    )


workflow = StateGraph(State)

members = [
    "Hypothesis",
    "Process",
    "Visualization",
    "Search",
    "Coder",
    "Report",
    "QualityReview",
    "Refiner",
]

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

hypothesis_agent = create_agent(
    llm=llm,
    tools=[collect_data, wikipedia, google_search, FireCrawl_scrape_webpages]
    + load_tools(
        ["arxiv"],
    ),
    system_message="""
As an esteemed expert in data analysis, your task is to formulate a set of research hypotheses and outline the steps to be taken based on the information table provided. 
Utilize statistics, machine learning, deep learning, and artificial intelligence in developing these hypotheses. 
Your hypotheses should be precise, achievable, professional, and innovative. 
To ensure the feasibility and uniqueness of your hypotheses, thoroughly investigate relevant information. 
For each hypothesis, include ample references to support your claims.

Upon analyzing the information table, you are required to:

1. Formulate research hypotheses that leverage statistics, machine learning, deep learning, and AI techniques.
2. Outline the steps involved in testing these hypotheses.
3. Verify the feasibility and uniqueness of each hypothesis through a comprehensive literature review.

At the conclusion of your analysis, present the complete research hypotheses, elaborate on their uniqueness and feasibility, and provide relevant references to support your assertions. 
Please answer in structured way to enhance readability.
Just answer a research hypothesis.
""",
    team_members=members,
    working_directory=WORKING_DIRECTORY,
)

process_agent = create_supervisor(
    power_llm,
    """
    You are a research supervisor responsible for overseeing and coordinating a comprehensive data analysis project, resulting in a complete and cohesive research report. Your primary tasks include:

    1. Validating and refining the research hypothesis to ensure it is clear, specific, and testable.
    2. Orchestrating a thorough data analysis process, with all code well-documented and reproducible.
    3. Compiling and refining a research report that includes:
        - Introduction
        - Hypothesis
        - Methodology
        - Results, accompanied by relevant visualizations
        - Discussion
        - Conclusion
        - References

    **Step-by-Step Process:**
    1. **Planning:** Define clear objectives and expected outcomes for each phase of the project.
    2. **Task Assignment:** Assign specific tasks to the appropriate agents ("Visualization," "Search," "Coder," "Report").
    3. **Review and Integration:** Critically review and integrate outputs from each agent, ensuring consistency, quality, and relevance.
    4. **Feedback:** Provide feedback and further instructions as needed to refine outputs.
    5. **Final Compilation:** Ensure all components are logically connected and meet high academic standards.

    **Agent Guidelines:**
    - **Visualization Agent:** Develop and explain data visualizations that effectively communicate key findings.
    - **Search Agent:** Collect and summarize relevant information, and compile a comprehensive list of references.
    - **Coder Agent:** Write and document efficient Python code for data analysis, ensuring that the code is clean and reproducible.
    - **Report Agent:** Draft, refine, and finalize the research report, integrating inputs from all agents and ensuring the narrative is clear and cohesive.

    **Workflow:**
    1. Plan the overall analysis and reporting process.
    2. Assign tasks to the appropriate agents and oversee their progress.
    3. Continuously review and integrate the outputs from each agent, ensuring that each contributes effectively to the final report.
    4. Adjust the analysis and reporting process based on emerging results and insights.
    5. Compile the final report, ensuring all sections are complete and well-integrated.

    **Completion Criteria:**
    Respond with "FINISH" only when:
    1. The hypothesis has been thoroughly tested and validated.
    2. The data analysis is complete, with all code documented and reproducible.
    3. All required visualizations have been created, properly labeled, and explained.
    4. The research report is comprehensive, logically structured, and includes all necessary sections.
    5. The reference list is complete and accurately cited.
    6. All components are cohesively integrated into a polished final report.

    Ensure that the final report delivers a clear, insightful analysis, addressing all aspects of the hypothesis and meeting the highest academic standards.
    """,
    ["Visualization", "Search", "Coder", "Report"],
)

visualization_agent = create_agent(
    llm,
    [read_document, execute_code, execute_command],
    """
    You are a data visualization expert tasked with creating insightful visual representations of data. Your primary responsibilities include:

    1. Designing appropriate visualizations that clearly communicate data trends and patterns.
    2. Selecting the most suitable chart types (e.g., bar charts, scatter plots, heatmaps) for different data types and analytical purposes.
    3. Providing executable Python code (using libraries such as matplotlib, seaborn, or plotly) that generates these visualizations.
    4. Including well-defined titles, axis labels, legends, and saving the visualizations as files.
    5. Offering brief but clear interpretations of the visual findings.

    **File Saving Guidelines:**
    - Save all visualizations as files with descriptive and meaningful filenames.
    - Ensure filenames are structured to easily identify the content (e.g., 'sales_trends_2024.png' for a sales trend chart).
    - Confirm that the saved files are organized in the working directory, making them easy for other agents to locate and use.

    **Constraints:**
    - Focus solely on visualization tasks; do not perform data analysis or preprocessing.
    - Ensure all visual elements are suitable for the target audience, with attention to color schemes and design principles.
    - Avoid over-complicating visualizations; aim for clarity and simplicity.
    """,
    members,
    WORKING_DIRECTORY,
)

code_agent = create_agent(
    power_llm,
    [read_document, execute_code, execute_command],
    """
    You are an expert Python programmer specializing in data processing and analysis. Your main responsibilities include:

    1. Writing clean, efficient Python code for data manipulation, cleaning, and transformation.
    2. Implementing statistical methods and machine learning algorithms as needed.
    3. Debugging and optimizing existing code for performance improvements.
    4. Adhering to PEP 8 standards and ensuring code readability with meaningful variable and function names.

    Constraints:
    - Focus solely on data processing tasks; do not generate visualizations or write non-Python code.
    - Provide only valid, executable Python code, including necessary comments for complex logic.
    - Avoid unnecessary complexity; prioritize readability and efficiency.
    """,
    members,
    WORKING_DIRECTORY,
)

searcher_agent = create_agent(
    llm,
    [collect_data, wikipedia, google_search, FireCrawl_scrape_webpages]
    + load_tools(
        ["arxiv"],
    ),
    """
    You are a skilled research assistant responsible for gathering and summarizing relevant information. Your main tasks include:

    1. Conducting thorough literature reviews using academic databases and reputable online sources.
    2. Summarizing key findings in a clear, concise manner.
    3. Providing citations for all sources, prioritizing peer-reviewed and academically reputable materials.

    Constraints:
    - Focus exclusively on information retrieval and summarization; do not engage in data analysis or processing.
    - Present information in an organized format, with clear attributions to sources.
    - Evaluate the credibility of sources and prioritize high-quality, reliable information.
    """,
    members,
    WORKING_DIRECTORY,
)

report_agent = create_agent(
    power_llm,
    [create_document, read_document, edit_document],
    """
    You are an experienced scientific writer tasked with drafting comprehensive research reports. Your primary duties include:

    1. Clearly stating the research hypothesis and objectives in the introduction.
    2. Detailing the methodology used, including data collection and analysis techniques.
    3. Structuring the report into coherent sections (e.g., Introduction, Methodology, Results, Discussion, Conclusion).
    4. Synthesizing information from various sources into a unified narrative.
    5. Integrating relevant data visualizations and ensuring they are appropriately referenced and explained.

    Constraints:
    - Focus solely on report writing; do not perform data analysis or create visualizations.
    - Maintain an objective, academic tone throughout the report.
    - Cite all sources using APA style and ensure that all findings are supported by evidence.
    """,
    members,
    WORKING_DIRECTORY,
)

quality_review_agent = create_agent(
    llm,
    [create_document, read_document, edit_document],
    """
    You are a meticulous quality control expert responsible for reviewing and ensuring the high standard of all research outputs. Your tasks include:

    1. Critically evaluating the content, methodology, and conclusions of research reports.
    2. Checking for consistency, accuracy, and clarity in all documents.
    3. Identifying areas that need improvement or further elaboration.
    4. Ensuring adherence to scientific writing standards and ethical guidelines.

    After your review, if revisions are needed, respond with 'REVISION' as a prefix, set needs_revision=True, and provide specific feedback on parts that need improvement. If no revisions are necessary, respond with 'CONTINUE' as a prefix and set needs_revision=False.
    """,
    members,
    WORKING_DIRECTORY,
)

note_agent = create_note_agent(
    json_llm,
    [read_document],
    """
    You are a meticulous research process note-taker. Your main responsibility is to observe, summarize, and document the actions and findings of the research team. Your tasks include:

    1. Observing and recording key activities, decisions, and discussions among team members.
    2. Summarizing complex information into clear, concise, and accurate notes.
    3. Organizing notes in a structured format that ensures easy retrieval and reference.
    4. Highlighting significant insights, breakthroughs, challenges, or any deviations from the research plan.
    5. Responding only in JSON format to ensure structured documentation.

    Your output should be well-organized and easy to integrate with other project documentation.
    """,
)

refiner_agent = create_agent(
    power_llm,
    [
        read_document,
        edit_document,
        create_document,
        collect_data,
        wikipedia,
        google_search,
        FireCrawl_scrape_webpages,
    ]
    + load_tools(
        ["arxiv"],
    ),
    """
    You are an expert AI report refiner tasked with optimizing and enhancing research reports. Your responsibilities include:

    1. Thoroughly reviewing the entire research report, focusing on content, structure, and readability.
    2. Identifying and emphasizing key findings, insights, and conclusions.
    3. Restructuring the report to improve clarity, coherence, and logical flow.
    4. Ensuring that all sections are well-integrated and support the primary research hypothesis.
    5. Condensing redundant or repetitive content while preserving essential details.
    6. Enhancing the overall readability, ensuring the report is engaging and impactful.

    Refinement Guidelines:
    - Maintain the scientific accuracy and integrity of the original content.
    - Ensure all critical points from the original report are preserved and clearly articulated.
    - Improve the logical progression of ideas and arguments.
    - Highlight the most significant results and their implications for the research hypothesis.
    - Ensure that the refined report aligns with the initial research objectives and hypothesis.

    After refining the report, submit it for final human review, ensuring it is ready for publication or presentation.
    """,
    members,
    WORKING_DIRECTORY,
)


def agent_node(state: State, agent: AgentExecutor, name: str) -> State:
    """
    Process an agent's action and update the state accordingly.
    """
    print(f"Processing agent: {name}")
    try:
        result = agent.invoke(state)
        print(f"Agent {name} result: {result}")

        output = (
            result["output"]
            if isinstance(result, dict) and "output" in result
            else str(result)
        )

        ai_message = AIMessage(content=output, name=name)
        state["messages"].append(ai_message)
        state["sender"] = name

        if name == "hypothesis_agent" and not state["hypothesis"]:
            state["hypothesis"] = ai_message
            print("Hypothesis updated")
        elif name == "process_agent":
            state["process_decision"] = ai_message
            print("Process decision updated")
        elif name == "visualization_agent":
            state["visualization_state"] = ai_message
            print("Visualization state updated")
        elif name == "searcher_agent":
            state["searcher_state"] = ai_message
            print("Searcher state updated")
        elif name == "report_agent":
            state["report_section"] = ai_message
            print("Report section updated")
        elif name == "quality_review_agent":
            state["quality_review"] = ai_message
            state["needs_revision"] = "revision needed" in output.lower()
            print(f"Quality review updated. Needs revision: {state['needs_revision']}")

        print(f"Agent {name} processing completed")
        return state
    except Exception as e:
        print(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}


def create_message(message: dict[str], name: str) -> BaseMessage:
    """
    Create a BaseMessage object based on the message type.
    """
    content = message.get("content", "")
    message_type = message.get("type", "").lower()

    print(f"Creating message of type {message_type} for {name}")
    return (
        HumanMessage(content=content)
        if message_type == "human"
        else AIMessage(content=content, name=name)
    )


def _create_error_state(
    state: State, error_message: AIMessage, name: str, error_type: str
) -> State:
    """
    Create an error state when an exception occurs.
    """
    print(f"Creating error state for {name}: {error_type}")
    error_state: State = {
        "messages": state.get("messages", []) + [error_message],
        "hypothesis": str(state.get("hypothesis", "")),
        "process": str(state.get("process", "")),
        "process_decision": str(state.get("process_decision", "")),
        "visualization_state": str(state.get("visualization_state", "")),
        "searcher_state": str(state.get("searcher_state", "")),
        "code_state": str(state.get("code_state", "")),
        "report_section": str(state.get("report_section", "")),
        "quality_review": str(state.get("quality_review", "")),
        "needs_revision": bool(state.get("needs_revision", False)),
        "sender": "note_agent",
    }
    return error_state


def note_agent_node(state: State, agent: AgentExecutor, name: str) -> State:
    """
    Process the note agent's action and update the entire state.
    """
    print(f"Processing note agent: {name}")
    try:
        current_messages = state.get("messages", [])

        head_messages, tail_messages = [], []

        if len(current_messages) > 6:
            head_messages = current_messages[:2]
            tail_messages = current_messages[-2:]
            state = {**state, "messages": current_messages[2:-2]}
            print("Trimmed messages for processing")

        result = agent.invoke(state)
        print(f"Note agent {name} result: {result}")
        output = (
            result["output"]
            if isinstance(result, dict) and "output" in result
            else str(result)
        )

        cleaned_output = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", output)
        parsed_output = json.loads(cleaned_output)
        print(f"Parsed output: {parsed_output}")

        new_messages = [
            create_message(msg, name) for msg in parsed_output.get("messages", [])
        ]

        messages = new_messages if new_messages else current_messages

        combined_messages = head_messages + messages + tail_messages

        updated_state: State = {
            "messages": combined_messages,
            "hypothesis": str(
                parsed_output.get("hypothesis", state.get("hypothesis", ""))
            ),
            "process": str(parsed_output.get("process", state.get("process", ""))),
            "process_decision": str(
                parsed_output.get("process_decision", state.get("process_decision", ""))
            ),
            "visualization_state": str(
                parsed_output.get(
                    "visualization_state", state.get("visualization_state", "")
                )
            ),
            "searcher_state": str(
                parsed_output.get("searcher_state", state.get("searcher_state", ""))
            ),
            "code_state": str(
                parsed_output.get("code_state", state.get("code_state", ""))
            ),
            "report_section": str(
                parsed_output.get("report_section", state.get("report_section", ""))
            ),
            "quality_review": str(
                parsed_output.get("quality_review", state.get("quality_review", ""))
            ),
            "needs_revision": bool(
                parsed_output.get("needs_revision", state.get("needs_revision", False))
            ),
            "sender": "note_agent",
        }

        print("Updated state successfully")
        return updated_state

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}", exc_info=True)
        return _create_error_state(
            state,
            AIMessage(content=f"Error parsing output: {output}", name=name),
            name,
            "JSON decode error",
        )

    except InternalServerError as e:
        print(f"OpenAI Internal Server Error: {e}", exc_info=True)
        return _create_error_state(
            state,
            AIMessage(content=f"OpenAI Error: {str(e)}", name=name),
            name,
            "OpenAI error",
        )

    except Exception as e:
        print(f"Unexpected error in note_agent_node: {e}", exc_info=True)
        return _create_error_state(
            state,
            AIMessage(content=f"Unexpected error: {str(e)}", name=name),
            name,
            "Unexpected error",
        )


def human_review_node(state: State) -> State:
    """
    Display current state to the user and update the state based on user input.
    Includes error handling for robustness.
    """
    try:
        print("Current research progress:")
        print(state)
        print("\nDo you need additional analysis or modifications?")

        while True:
            user_input = input(
                "Enter 'yes' to continue analysis, or 'no' to end the research: "
            ).lower()
            if user_input in ["yes", "no"]:
                break
            print("Invalid input. Please enter 'yes' or 'no'.")

        if user_input == "yes":
            while True:
                additional_request = input(
                    "Please enter your additional analysis request: "
                ).strip()
                if additional_request:
                    state["messages"].append(HumanMessage(content=additional_request))
                    state["needs_revision"] = True
                    break
                print("Request cannot be empty. Please try again.")
        else:
            state["needs_revision"] = False

        state["sender"] = "human"
        print("Human review completed successfully.")
        return state

    except KeyboardInterrupt:
        print("Human review interrupted by user.")
        return None

    except Exception as e:
        print(f"An error occurred during human review: {str(e)}", exc_info=True)
        return None


def refiner_node(state: State, agent: AgentExecutor, name: str) -> State:
    """
    Read MD file contents and PNG file names from the specified storage path,
    add them as report materials to a new message,
    then process with the agent and update the original state.
    If token limit is exceeded, use only MD file names instead of full content.
    """
    try:
        # Get storage path
        storage_path = Path(os.getenv("STORAGE_PATH", "./data_analysis/"))

        # Collect materials
        materials = []
        md_files = list(storage_path.glob("*.md"))
        png_files = list(storage_path.glob("*.png"))

        # Process MD files
        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                materials.append(f"MD file '{md_file.name}':\n{f.read()}")

        # Process PNG files
        materials.extend(f"PNG file: '{png_file.name}'" for png_file in png_files)

        # Combine materials
        combined_materials = "\n\n".join(materials)
        report_content = f"Report materials:\n{combined_materials}"

        # Create refiner state
        refiner_state = state.copy()
        refiner_state["messages"] = [BaseMessage(content=report_content)]

        try:
            # Attempt to invoke agent with full content
            result = agent.invoke(refiner_state)
        except Exception as token_error:
            # If token limit is exceeded, retry with only MD file names
            print("Token limit exceeded. Retrying with MD file names only.")
            md_file_names = [f"MD file: '{md_file.name}'" for md_file in md_files]
            png_file_names = [f"PNG file: '{png_file.name}'" for png_file in png_files]

            simplified_materials = "\n".join(md_file_names + png_file_names)
            simplified_report_content = (
                f"Report materials (file names only):\n{simplified_materials}"
            )

            refiner_state["messages"] = [BaseMessage(content=simplified_report_content)]
            result = agent.invoke(refiner_state)

        # Update original state
        state["messages"].append(AIMessage(content=result))
        state["sender"] = name

        print("Refiner node processing completed")
        return state
    except Exception as e:
        print(f"Error occurred while processing refiner node: {str(e)}", exc_info=True)
        state["messages"].append(AIMessage(content=f"Error: {str(e)}", name=name))
        return state


def human_choice_node(state: State) -> State:
    """
    Handle human input to choose the next step in the process.
    If regenerating hypothesis, prompt for specific areas to modify.
    """
    print("Prompting for human choice")
    print("Please choose the next step:")
    print("1. Regenerate hypothesis")
    print("2. Continue the research process")

    while True:
        choice = input("Please enter your choice (1 or 2): ")
        if choice in ["1", "2"]:
            break
        print(f"Invalid input received: {choice}")
        print("Invalid input, please try again.")

    if choice == "1":
        modification_areas = input(
            "Please specify which parts of the hypothesis you want to modify: "
        )
        content = f"Regenerate hypothesis. Areas to modify: {modification_areas}"
        state["hypothesis"] = ""
        state["modification_areas"] = modification_areas
        print("Hypothesis cleared for regeneration")
        print(f"Areas to modify: {modification_areas}")
    else:
        content = "Continue the research process"
        state["process"] = "Continue the research process"
        print("Continuing research process")

    human_message = HumanMessage(content=content)

    state["messages"].append(human_message)
    state["sender"] = "human"

    print("Human choice processed")
    return state


NodeType = Literal[
    "Visualization",
    "Search",
    "Coder",
    "Report",
    "Process",
    "NoteTaker",
    "Hypothesis",
    "QualityReview",
]
ProcessNodeType = Literal[
    "Coder", "Search", "Visualization", "Report", "Process", "Refiner"
]


def hypothesis_router(state: State) -> NodeType:
    """
    Route based on the presence of a hypothesis in the state.

    Args:
    state (State): The current state of the system.

    Returns:
    NodeType: 'Hypothesis' if no hypothesis exists, otherwise 'Process'.
    """
    print("Entering hypothesis_router")
    hypothesis = state.get("hypothesis")

    if isinstance(hypothesis, AIMessage):
        hypothesis_content = hypothesis.content
        print("Hypothesis is an AIMessage")
    elif isinstance(hypothesis, str):
        hypothesis_content = hypothesis
        print("Hypothesis is a string")
    else:
        hypothesis_content = ""
        print(f"Unexpected hypothesis type: {type(hypothesis)}")

    result = "Hypothesis" if not hypothesis_content.strip() else "Process"
    print(f"hypothesis_router decision: {result}")
    return result


def QualityReview_router(state: State) -> NodeType:
    """
    Route based on the quality review outcome and process decision.

    Args:
    state (State): The current state of the system.

    Returns:
    NodeType: The next node to route to based on the quality review and process decision.
    """
    print("Entering QualityReview_router")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    # Check if revision is needed
    if (last_message and "REVISION" in str(last_message.content)) or state.get(
        "needs_revision", False
    ):
        previous_node = state.get("last_sender", "")
        revision_routes = {
            "Visualization": "Visualization",
            "Search": "Search",
            "Coder": "Coder",
            "Report": "Report",
        }
        result = revision_routes.get(previous_node, "NoteTaker")
        print(f"Revision needed. Routing to: {result}")
        return result

    else:
        return "NoteTaker"


def process_router(state: State) -> ProcessNodeType:
    """
    Route based on the process decision in the state.

    Args:
    state (State): The current state of the system.

    Returns:
    ProcessNodeType: The next process node to route to based on the process decision.
    """
    print("Entering process_router")
    process_decision = state.get("process_decision", "")

    # Handle AIMessage object
    if isinstance(process_decision, AIMessage):
        print("Process decision is an AIMessage")
        try:
            # Attempt to parse JSON in content
            decision_dict = json.loads(process_decision.content.replace("'", '"'))
            process_decision = decision_dict.get("next", "")
            print(f"Parsed process decision from JSON: {process_decision}")
        except json.JSONDecodeError:
            # If JSON parsing fails, use content directly
            process_decision = process_decision.content
            print("Failed to parse process decision as JSON. Using content directly.")
    elif isinstance(process_decision, dict):
        process_decision = process_decision.get("next", "")
        print(
            f"Process decision is a dictionary. Using 'next' value: {process_decision}"
        )
    elif not isinstance(process_decision, str):
        process_decision = str(process_decision)
        print(
            f"Unexpected process decision type. Converting to string: {process_decision}"
        )

    # Define valid decisions
    valid_decisions = {"Coder", "Search", "Visualization", "Report"}

    if process_decision in valid_decisions:
        print(f"Valid process decision: {process_decision}")
        return process_decision

    if process_decision == "FINISH":
        print("Process decision is FINISH. Ending process.")
        return "Refiner"

    # If process_decision is empty or not a valid decision, return "Process"
    if not process_decision or process_decision not in valid_decisions:
        print(
            f"Invalid or empty process decision: {process_decision}. Defaulting to 'Process'."
        )
        return "Process"

    # Default to "Process"
    print("Defaulting to 'Process'")
    return "Process"


workflow.add_node(
    "Hypothesis", lambda state: agent_node(state, hypothesis_agent, "hypothesis_agent")
)
workflow.add_node(
    "Process", lambda state: agent_node(state, process_agent, "process_agent")
)
workflow.add_node(
    "Visualization",
    lambda state: agent_node(state, visualization_agent, "visualization_agent"),
)
workflow.add_node(
    "Search", lambda state: agent_node(state, searcher_agent, "searcher_agent")
)
workflow.add_node("Coder", lambda state: agent_node(state, code_agent, "code_agent"))
workflow.add_node(
    "Report", lambda state: agent_node(state, report_agent, "report_agent")
)
workflow.add_node(
    "QualityReview",
    lambda state: agent_node(state, quality_review_agent, "quality_review_agent"),
)
workflow.add_node(
    "NoteTaker", lambda state: note_agent_node(state, note_agent, "note_agent")
)
workflow.add_node("HumanChoice", human_choice_node)
workflow.add_node("HumanReview", human_review_node)
workflow.add_node(
    "Refiner", lambda state: refiner_node(state, refiner_agent, "refiner_agent")
)

workflow.add_edge("Hypothesis", "HumanChoice")
workflow.add_conditional_edges(
    "HumanChoice", hypothesis_router, {"Hypothesis": "Hypothesis", "Process": "Process"}
)

workflow.add_conditional_edges(
    "Process",
    process_router,
    {
        "Coder": "Coder",
        "Search": "Search",
        "Visualization": "Visualization",
        "Report": "Report",
        "Process": "Process",
        "Refiner": "Refiner",
    },
)

for member in ["Visualization", "Search", "Coder", "Report"]:
    workflow.add_edge(member, "QualityReview")

workflow.add_conditional_edges(
    "QualityReview",
    QualityReview_router,
    {
        "Visualization": "Visualization",
        "Search": "Search",
        "Coder": "Coder",
        "Report": "Report",
        "NoteTaker": "NoteTaker",
    },
)
workflow.add_edge("NoteTaker", "Process")

workflow.add_edge("Refiner", "HumanReview")

# Add an edge from HumanReview to Process
workflow.add_conditional_edges(
    "HumanReview",
    lambda state: "Process" if state and state.get("needs_revision", False) else "END",
    {"Process": "Process", "END": END},
)

workflow.add_edge(START, "Hypothesis")

memory = MemorySaver()

graph = workflow.compile()

userInput = """
data_path:OnlineSalesData.csv
Use machine learning to perform data analysis and write complete graphical reports
"""

events = graph.stream(
    {
        "messages": [
            HumanMessage(content=userInput),
        ],
        "hypothesis": "",
        "process_decision": "",
        "process": "",
        "visualization_state": "",
        "searcher_state": "",
        "code_state": "",
        "report_section": "",
        "quality_review": "",
        "needs_revision": False,
        "last_sender": "",
    },
    {"configurable": {"thread_id": "1"}, "recursion_limit": 3000},
    stream_mode="values",
    debug=False,
)


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message, end="", flush=True)
        else:
            message.pretty_print()


print_stream(events)
