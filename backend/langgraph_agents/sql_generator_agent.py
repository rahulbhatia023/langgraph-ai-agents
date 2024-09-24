import operator
import os
from typing import Dict, Any, List, Annotated

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

load_dotenv()

endpoint_url = os.getenv("SQLITE_DB_ENDPOINT_URL")

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class InputState(MessagesState):
    question: str
    uuid: str
    parsed_question: Dict[str, Any]
    unique_nouns: List[str]
    sql_query: str
    results: List[Any]


class OutputState(MessagesState):
    parsed_question: Dict[str, Any]
    unique_nouns: List[str]
    sql_query: str
    sql_valid: bool
    sql_issues: str
    results: List[Any]
    answer: Annotated[str, operator.add]
    error: str


def invoke_llm(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def get_schema(uuid: str) -> str:
    """Retrieve the database schema."""
    try:
        response = requests.get(f"{endpoint_url}/get-schema/{uuid}")
        response.raise_for_status()
        return response.json()["schema"]
    except requests.RequestException as e:
        raise Exception(f"Error fetching schema: {str(e)}")


def execute_query(uuid: str, query: str) -> List[Any]:
    """Execute SQL query on the remote database and return results."""
    try:
        response = requests.post(
            f"{endpoint_url}/execute-query", json={"uuid": uuid, "query": query}
        )
        response.raise_for_status()
        return response.json()["results"]
    except requests.RequestException as e:
        raise Exception(f"Error executing query: {str(e)}")


def ask_question(state):
    pass


def parse_question(state):
    """Parse user question and identify relevant tables and columns."""
    question = state["question"]
    schema = get_schema(state["uuid"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a data analyst that can help summarize SQL tables and parse user questions about a database. 
                Given the question and database schema, identify the relevant tables and columns. 
                        If the question is not relevant to the database or if there is not enough information to answer the question, set is_relevant to false.
                        
                        Your response should be in the following JSON format:
                        {{
                            "is_relevant": boolean,
                            "relevant_tables": [
                                {{
                                    "table_name": string,
                                    "columns": [string],
                                    "noun_columns": [string]
                                }}
                            ]
                        }}

                        The "noun_columns" field should contain only the columns that are relevant to the question and contain nouns or names, 
                        for example, the column "Artist name" contains nouns relevant to the question "What are the top selling artists?", but the column "Artist ID" is not relevant because it does not contain a noun. 
                        Do not include columns that contain numbers.""",
            ),
            (
                "human",
                "===Database schema:\n{schema}\n\n===User question:\n{question}\n\nIdentify relevant tables and columns:",
            ),
        ]
    )

    output_parser = JsonOutputParser()

    response = llm.invoke(
        prompt.format_messages(schema=schema, question=question)
    ).content

    parsed_response = output_parser.parse(response)

    return {"parsed_question": parsed_response}


def get_unique_nouns(state):
    """Find unique nouns in relevant tables and columns."""
    parsed_question = state["parsed_question"]

    if not parsed_question["is_relevant"]:
        return {"unique_nouns": []}

    unique_nouns = set()
    for table_info in parsed_question["relevant_tables"]:
        table_name = table_info["table_name"]
        noun_columns = table_info["noun_columns"]

        if noun_columns:
            column_names = ", ".join(f"`{col}`" for col in noun_columns)
            query = f"SELECT DISTINCT {column_names} FROM `{table_name}`"
            results = execute_query(state["uuid"], query)
            for row in results:
                unique_nouns.update(str(value) for value in row if value)

    return {"unique_nouns": list(unique_nouns)}


def generate_sql(state: dict) -> dict:
    """Generate SQL query based on parsed question and unique nouns."""
    question = state["question"]
    parsed_question = state["parsed_question"]
    unique_nouns = state["unique_nouns"]

    if not parsed_question["is_relevant"]:
        return {"sql_query": "NOT_RELEVANT", "is_relevant": False}

    schema = get_schema(state["uuid"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI assistant that generates SQL queries based on user questions, database schema, and unique nouns found in the relevant tables. Generate a valid SQL query to answer the user's question.
                
                If there is not enough information to write a SQL query, respond with "NOT_ENOUGH_INFO".
                
                Here are some examples:
                
                1. What is the top selling product?
                Answer: SELECT product_name, SUM(quantity) as total_quantity FROM sales WHERE product_name IS NOT NULL AND quantity IS NOT NULL AND product_name != "" AND quantity != "" AND product_name != "N/A" AND quantity != "N/A" GROUP BY product_name ORDER BY total_quantity DESC LIMIT 1
                
                2. What is the total revenue for each product?
                Answer: SELECT \`product name\`, SUM(quantity * price) as total_revenue FROM sales WHERE \`product name\` IS NOT NULL AND quantity IS NOT NULL AND price IS NOT NULL AND \`product name\` != "" AND quantity != "" AND price != "" AND \`product name\` != "N/A" AND quantity != "N/A" AND price != "N/A" GROUP BY \`product name\`  ORDER BY total_revenue DESC
                
                3. What is the market share of each product?
                Answer: SELECT \`product name\`, SUM(quantity) * 100.0 / (SELECT SUM(quantity) FROM sa  les) as market_share FROM sales WHERE \`product name\` IS NOT NULL AND quantity IS NOT NULL AND \`product name\` != "" AND quantity != "" AND \`product name\` != "N/A" AND quantity != "N/A" GROUP BY \`product name\`  ORDER BY market_share DESC
                
                4. Plot the distribution of income over time
                Answer: SELECT income, COUNT(*) as count FROM users WHERE income IS NOT NULL AND income != "" AND income != "N/A" GROUP BY income
                
                THE RESULTS SHOULD ONLY BE IN THE FOLLOWING FORMAT, SO MAKE SURE TO ONLY GIVE TWO OR THREE COLUMNS:
                [[x, y]]
                or 
                [[label, x, y]]
                
                For questions like "plot a distribution of the fares for men and women", count the frequency of each fare and plot it. The x axis should be the fare and the y axis should be the count of people who paid that fare.
                SKIP ALL ROWS WHERE ANY COLUMN IS NULL or "N/A" or "".
                Just give the query string. Do not format it. Make sure to use the correct spellings of nouns as provided in the unique nouns list. All the table and column names should be enclosed in backticks.
                """,
            ),
            (
                "human",
                """===Database schema:
                {schema}
                
                ===User question:
                {question}
                
                ===Relevant tables and columns:
                {parsed_question}
                
                ===Unique nouns in relevant tables:
                {unique_nouns}
                
                Generate SQL query string""",
            ),
        ]
    )

    response = llm.invoke(
        prompt.format_messages(
            schema=schema,
            question=question,
            parsed_question=parsed_question,
            unique_nouns=unique_nouns,
        )
    ).content

    if response.strip() == "NOT_ENOUGH_INFO":
        return {"sql_query": "NOT_RELEVANT"}
    else:
        return {"sql_query": response}


def validate_and_fix_sql(state: dict) -> dict:
    """Validate and fix the generated SQL query."""
    sql_query = state["sql_query"]

    if sql_query == "NOT_RELEVANT":
        return {"sql_query": "NOT_RELEVANT", "sql_valid": False}

    schema = get_schema(state["uuid"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are an AI assistant that validates and fixes SQL queries. Your task is to:
                    1. Check if the SQL query is valid.
                    2. Ensure all table and column names are correctly spelled and exist in the schema. All the table and column names should be enclosed in backticks.
                    3. If there are any issues, fix them and provide the corrected SQL query.
                    4. If no issues are found, return the original query.
                    
                    Respond in JSON format with the following structure. Only respond with the JSON:
                    {{
                        "valid": boolean,
                        "issues": string or null,
                        "corrected_query": string
                    }}
                    """,
            ),
            (
                "human",
                """===Database schema:
                    {schema}
                    
                    ===Generated SQL query:
                    {sql_query}
                    
                    Respond in JSON format with the following structure. Only respond with the JSON:
                    {{
                        "valid": boolean,
                        "issues": string or null,
                        "corrected_query": string
                    }}
                    
                    For example:
                    1. {{
                        "valid": true,
                        "issues": null,
                        "corrected_query": "None"
                    }}
                    
                    2. {{
                        "valid": false,
                        "issues": "Column USERS does not exist",
                        "corrected_query": "SELECT * FROM \`users\` WHERE age > 25"
                    }}
                    
                    3. {{
                        "valid": false,
                        "issues": "Column names and table names should be enclosed in backticks if they contain spaces or special characters",
                        "corrected_query": "SELECT * FROM \`gross income\` WHERE \`age\` > 25"
                    }}
                    
                    """,
            ),
        ]
    )

    output_parser = JsonOutputParser()

    response = llm.invoke(
        prompt.format_messages(schema=schema, sql_query=sql_query)
    ).content

    result = output_parser.parse(response)

    if result["valid"] and result["issues"] is None:
        return {"sql_query": sql_query, "sql_valid": True}
    else:
        return {
            "sql_query": result["corrected_query"],
            "sql_valid": result["valid"],
            "sql_issues": result["issues"],
        }


def execute_sql(state: dict) -> dict:
    """Execute SQL query and return results."""
    query = state["sql_query"]
    uuid = state["uuid"]

    if query == "NOT_RELEVANT":
        return {"results": "NOT_RELEVANT"}

    try:
        results = execute_query(uuid, query)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


def format_results(state: dict) -> dict:
    """Format query results into a human-readable response."""
    question = state["question"]
    results = state["results"]

    if results == "NOT_RELEVANT":
        return {"answer": "Sorry, I can only give answers relevant to the database."}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant that formats database query results into a human-readable response. Give a conclusion to the user's question based on the query results. Do not give the answer in markdown format. Only give the answer in one line.",
            ),
            (
                "human",
                "User question: {question}\n\nQuery results: {results}\n\nFormatted response:",
            ),
        ]
    )

    response = llm.invoke(
        prompt.format_messages(question=question, results=results)
    ).content

    final_response_prompt = """
    Below are the given sql query, query results and final response
    Present all these in a pretty manner to the user.
    
    sql query: {query}
    query results: {results}
    final response: {response}
    """

    final_response = llm.invoke(final_response_prompt.format(query=state["sql_query"], results=state["results"], response=response)).content

    return {"messages": [AIMessage(content=final_response)]}


workflow = StateGraph(input=InputState, output=OutputState)

workflow.add_node("agent", invoke_llm)
workflow.add_node("ask_question", ask_question)
workflow.add_node("parse_question", parse_question)
workflow.add_node("get_unique_nouns", get_unique_nouns)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("validate_and_fix_sql", validate_and_fix_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("format_results", format_results)

# Define edges
workflow.add_edge(START, "agent")
workflow.add_edge("agent", "ask_question")
workflow.add_edge("ask_question", "parse_question")
workflow.add_edge("parse_question", "get_unique_nouns")
workflow.add_edge("get_unique_nouns", "generate_sql")
workflow.add_edge("generate_sql", "validate_and_fix_sql")
workflow.add_edge("validate_and_fix_sql", "execute_sql")
workflow.add_edge("execute_sql", "format_results")
workflow.add_edge("format_results", END)

agent = workflow.compile(checkpointer=MemorySaver(), interrupt_before=["ask_question"])


def get_agent():
    return agent


"""
question = "What is the market share of products?"
uuid = "921c838c-541d-4361-8c96-70cb23abd9f5"

parsed_question = parse_question(
    {
        "question": question,
        "uuid": uuid,
    }
)["parsed_question"]
print(parsed_question)
# {'is_relevant': True, 'relevant_tables': [{'table_name': 'supermarket_sales - Sheet1', 'columns': ['Product line', 'Quantity', 'Total'], 'noun_columns': ['Product line']}]}

unique_nouns = get_unique_nouns({"parsed_question": parsed_question, "uuid": uuid})[
    "unique_nouns"
]
print(unique_nouns)
# ['Food and beverages', 'Fashion accessories', 'Electronic accessories', 'Health and beauty', 'Home and lifestyle', 'Sports and travel']

sql_query = generate_sql(
    {
        "question": question,
        "parsed_question": parsed_question,
        "unique_nouns": unique_nouns,
        "uuid": uuid,
    }
)["sql_query"]
print(sql_query)
# 'SELECT `Product line`, SUM(`Quantity`) * 100.0 / (SELECT SUM(`Quantity`) FROM `supermarket_sales - Sheet1`) as market_share FROM `supermarket_sales - Sheet1` WHERE `Product line` IS NOT NULL AND `Quantity` IS NOT NULL AND `Product line` != "" AND `Quantity` != "" AND `Product line` != "N/A" AND `Quantity` != "N/A" GROUP BY `Product line` ORDER BY market_share DESC'

sql_valid = validate_and_fix_sql({"sql_query": sql_query, "uuid": uuid})["sql_valid"]
print(sql_valid) # True

results = execute_sql({"sql_query": sql_query, "uuid": uuid})["results"]
print(results)
# [['Health and beauty', 22.164948453608247], ['Sports and travel', 19.415807560137456], ['Food and beverages', 17.52577319587629], ['Electronic accessories', 15.97938144329897], ['Home and lifestyle', 13.573883161512027], ['Fashion accessories', 11.34020618556701]]

answer = format_results({"question": question, "results": results})["answer"]
print(answer)
# The market share of products is as follows: Health and beauty (22.16%), Sports and travel (19.42%), Food and beverages (17.53%), Electronic accessories (15.98%), Home and lifestyle (13.57%), and Fashion accessories (11.34%).
"""
