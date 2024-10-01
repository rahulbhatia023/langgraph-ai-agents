import json
import operator
import os
from typing import Annotated, Union, List, TypedDict

import ollama
import praw
from dotenv import load_dotenv
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from praw.models import Comment
from pydantic.v1 import BaseModel
from semantic_router.utils.function_call import FunctionSchema

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="MyBot",
)

"""
We'll be pulling in submission threads from Reddit that include user's restaurant recommendations (or just other info we search for). 

From the submission threads we need:
- Submission title
- Submission first text / description
- A few of the top voted comments

To organize this information we can create a pydantic class to structure the needed data:
"""


class Rec(BaseModel):
    title: str
    description: str
    comments: list[str]

    def __str__(self):
        return f"\n\n**Title**:\n\n{self.title}\n\n\n**Description**:\n\n{self.description}\n\n\n**Comments:**\n\n{'\n\n'.join(self.comments)}\n\n ---------------------------"


def search(query: str) -> list[Rec]:
    """Provides access to search reddit. You can use this tool to find restaurants.
    Best results can be found by providing as much context as possible, including
    location, cuisine, and the fact that you're looking for a restaurant, cafe,
    etc.
    """

    results = reddit.subreddit("all").search(query)
    recs = []
    for submission in results:
        title = submission.title
        description = submission.selftext
        comments = []
        for comment in submission.comments.list():
            if isinstance(comment, Comment) and comment.ups >= 10:
                author = comment.author.name if comment.author else "unknown"
                comments.append(f"{author} (upvotes: {comment.ups}): {comment.body}")
        comments = comments[:3]
        if len(comments) == 3:
            recs.append(Rec(title=title, description=description, comments=comments))
        if len(recs) == 3:
            break
    return recs


def final_answer(answer: str):
    """Returns a natural language response to the user"""
    return {"answer": answer}


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    output: dict[str, Union[str, List[str]]]
    messages: Annotated[list[dict], operator.add]


"""
Alongside our system prompt, we must also pass Ollama the schema of our functions for tool calls. 
Tool calling is a relatively new feature in Ollama and is used by providing function schemas to the tools parameter when calling our LLM.
We use FunctionSchema object with to_ollama from semantic-router to transform our functions into correctly formatted schemas.
"""

search_schema = FunctionSchema(search).to_ollama()
search_schema["function"]["parameters"]["properties"]["query"]["description"] = None

final_answer_schema = FunctionSchema(final_answer).to_ollama()
for key in final_answer_schema["function"]["parameters"]["properties"].keys():
    final_answer_schema["function"]["parameters"]["properties"][key][
        "description"
    ] = None

system_message = """
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

Your goal is to provide the user with the best possible answer.

Note, when using a tool, you provide the tool name and the arguments to use
in JSON format. For each call, you MUST ONLY use one tool AND the response
format must ALWAYS be in the pattern:

```json
{
"name": "<tool_name>",
    "parameters": {"<tool_input_key>": <tool_input_value>}
}
```

Remember, NEVER use the search tool more than 3x as that can trigger
the nuclear annihilation system.

After using the search tool you must summarize your findings with the
final_answer tool and always answer based on the context provided.
"""


def get_system_tools_prompt(system_prompt: str, tools: list[dict]):
    tools_str = "\n".join([str(tool) for tool in tools])
    return f"{system_prompt}\n\n" f"You may use the following tools:\n{tools_str}"


class AgentAction(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str | None = None

    @classmethod
    def from_ollama(cls, ollama_response: dict):
        try:
            output = json.loads(ollama_response["message"]["content"])
            return cls(
                tool_name=output["name"],
                tool_input=output["parameters"],
            )
        except Exception as e:
            print(f"Error parsing ollama response:\n{ollama_response}\n")
            raise e

    def __str__(self):
        text = f"Tool: {self.tool_name}\nInput: {self.tool_input}"
        if self.tool_output is not None:
            text += f"\nOutput: {self.tool_output}"
        return text


"""
Now we just need to wrap this with the ability to contain chat history and the agent scratchpad — before adding everything into our graph.

For our agent actions, we will be converting them into fake back-and-forth messages between the assistant and user. For example:

AgentAction(
    tool_name="xyz",
    tool_input={"query": "something cool"},
    tool_output="A fascinating tidbit of information"
)
Would become:

[
    {"role": "assistant", "content": "{'name': 'xyz', 'parameters': {'query': 'something cool'}"},
    {"role": "user", "content": "A fascinating tidbit of information"}
]
We will make this happen with an action_to_message function:
"""


def action_to_message(action: AgentAction):
    assistant_message = {
        "role": "assistant",
        "content": json.dumps(
            {"name": action.tool_name, "parameters": action.tool_input}
        ),
    }
    user_message = {"role": "user", "content": action.tool_output}
    return [assistant_message, user_message]


def create_scratchpad(intermediate_steps: list[AgentAction]):
    intermediate_steps = [
        action for action in intermediate_steps if action.tool_output is not None
    ]
    scratch_pad_messages = []
    for action in intermediate_steps:
        scratch_pad_messages.extend(action_to_message(action))
    return scratch_pad_messages


def call_llm(
    user_input: str, chat_history: list[dict], intermediate_steps: list[AgentAction]
) -> AgentAction:
    scratchpad = create_scratchpad(intermediate_steps)

    if scratchpad:
        scratchpad += [
            {
                "role": "user",
                "content": (
                    f"Please continue, as a reminder my query was '{user_input}'. "
                    "Only answer to the original query, and nothing else — but use the "
                    "information I provided to you to do so. Provide as much "
                    "information as possible in the `answer` field of the final_answer tool"
                ),
            }
        ]
        tools_used = [action.tool_name for action in intermediate_steps]
        tools = []
        if "search" in tools_used:
            tools = [final_answer_schema]
            scratchpad[-1]["content"] = "You must now use the final_answer tool."
        else:
            tools = [search_schema, final_answer_schema]
    else:
        tools = [search_schema, final_answer_schema]

    messages = [
        {
            "role": "system",
            "content": get_system_tools_prompt(
                system_prompt=system_message, tools=tools
            ),
        },
        *chat_history,
        {"role": "user", "content": user_input},
        *scratchpad,
    ]

    ollama_response = ollama.chat(
        model="llama3.1:8b",
        messages=messages,
        format="json",
    )

    return AgentAction.from_ollama(ollama_response)


"""
We have defined the different logical components of our graph, but we need to execute them in a langgraph-friendly manner.
For that they must consume our AgentState and return modifications to that state. 
We will do this for all of our components via three functions:

`agent` will handle running our LLM.
`router` will handle the routing between our agent and tools.
`run_tool` will handle running our tool functions.
"""


def oracle(state):
    response = call_llm(
        user_input=state["input"],
        chat_history=state["chat_history"],
        intermediate_steps=state["intermediate_steps"],
    )

    return {
        "messages": [
            {
                "role": "assistant",
                "content": response.tool_output if response.tool_output else "",
            }
        ],
        "intermediate_steps": [response],
    }


def router(state):
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool_name
    else:
        return "final_answer"


tool_str_to_func = {"search": search, "final_answer": final_answer}


def run_tool(state):
    tool_name = state["intermediate_steps"][-1].tool_name
    tool_input = state["intermediate_steps"][-1].tool_input
    tool_output = tool_str_to_func[tool_name](**tool_input)

    action = AgentAction(
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=str(tool_output),
    )

    if tool_name == "final_answer":
        return {
            "messages": [{"role": "assistant", "content": tool_output["answer"]}],
            "output": tool_output,
        }
    else:
        recommendations = ""
        for rec in tool_output:
            recommendations += str(rec) + "\n\n"

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"""
                    Here are top 3 discussions with top 3 comments from which I would be analyzing my final response:
                    
                    {recommendations}""",
                }
            ],
            "intermediate_steps": [action],
        }


def user_input(state):
    last_message = state["messages"][-1]
    return {"input": last_message["content"]}


graph = StateGraph(AgentState)

graph.add_node("user_input", user_input)
graph.add_node("oracle", oracle)
graph.add_node("search", run_tool)
graph.add_node("final_answer", run_tool)

graph.add_edge(START, "user_input")
graph.add_edge("user_input", "oracle")
graph.add_conditional_edges(
    source="oracle",
    path=router,
    path_map={"search": "search", "final_answer": "final_answer"},
)

for tool in [search_schema, final_answer_schema]:
    tool_name = tool["function"]["name"]
    if tool_name != "final_answer":
        graph.add_edge(tool_name, "oracle")

graph.add_edge("final_answer", END)

agent = graph.compile(interrupt_before=["user_input"], checkpointer=MemorySaver())