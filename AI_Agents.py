import streamlit as st

page_header = "Welcome to the World of AI Agents"

font_family = "monospace"

background_color = "#0D1117"
text_color = "#58A6FF"
accent_color = "#2EA043"
card_bg_color = "#161B22"

page_header_style = f"""
    font-family: {font_family};
    color: {text_color};
    text-align: center;
"""
vertical_space_div_style = f"margin-top: 40px;margin-bottom: 40px;"
custom_card_div_style = f"""
    background-color: {card_bg_color};
    border: 1px solid {accent_color};
    border-radius: 5px;
    padding: 20px;
    height: 500px;
    margin-bottom: 30px;
    margin-left: 10px;
    font-family: {font_family};
"""
custom_card_title_style = f"""
    color: {accent_color};
    font-family: {font_family};
    font-weight: 600;
    margin-bottom: 10px;
    text-align: center;
"""
custom_card_points_style = f"font-family: {font_family};font-size: 14px;color: {text_color};"


def custom_card(title, description):
    return f"""
    <div style='{custom_card_div_style}'>
        <h4 style='{custom_card_title_style}'>{title}</h4>
        <p style='{custom_card_points_style}'>{"<br><br>".join([f"> {point}" for point in description])}</p>
    </div>
    """


st.set_page_config(page_title="AI Agents", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap');

    .stApp {
        background-color: #0D1117;
        font-family: 'JetBrains Mono';
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<h1 style='{page_header_style}'>{page_header}</h1>", unsafe_allow_html=True
)

st.markdown(f"<div style='{vertical_space_div_style}' />", unsafe_allow_html=True)

container = st.container()

col11, col12, col13 = container.columns(3)

with col11:
    st.markdown(
        custom_card(
            title="Data Query Assistant",
            description=[
                "Bridges the gap between natural language questions and datasets",
                "Allows users to upload SQLite databases or CSV files",
                "Translates user questions into SQL queries",
                "Executes SQL queries on the provided dataset",
                "Formats query results into human-readable responses",
            ],
        ),
        unsafe_allow_html=True,
    )

with col12:
    st.markdown(
        custom_card(
            title="Financial Assistant",
            description=[
                "Integrates a stock market API for real-time stock data and financial insights",
                "Incorporates the Tavily search API for broader web searches",
                "Provides a comprehensive tool for financial data and information retrieval",
                "Users should conduct their own research or consult a financial advisor before making decisions",
            ],
        ),
        unsafe_allow_html=True,
    )

with col13:
    st.markdown(
        custom_card(
            title="Podcast Script Writer",
            description=[
                "Generates content for podcasts based on provided topics",
                "Uses extensive web-based tool searches to augment information for the topic",
                "Reduces workload for podcast creators by providing structured and relevant content",
                "Combines advanced AI models and graph-based workflows",
            ],
        ),
        unsafe_allow_html=True,
    )

col21, col22, col23 = container.columns(3)

with col21:
    st.markdown(
        custom_card(
            title="Python and React Assistant",
            description=[
                "AI-powered assistant with Python execution capabilities",
                "Integrates real-time React component rendering",
                "Offers a comprehensive environment for data analysis",
                "Supports dynamic data visualization",
                "Facilitates interactive web development",
            ],
        ),
        unsafe_allow_html=True,
    )

with col22:
    st.markdown(
        custom_card(
            title="Reddit Search",
            description=[
                "Searches Reddit for relevant answers to user queries",
                "Scans multiple subreddits for comprehensive results",
                "Intelligently analyzes and filters out irrelevant responses",
                "Consolidates insights from various Reddit communities",
                "Provides users with accurate and curated information",
            ],
        ),
        unsafe_allow_html=True,
    )

with col23:
    st.markdown(
        custom_card(
            title="Research Analyst",
            description=[
                "Customizes the research process with AI analysts",
                "Assembles analysts based on user-selected sources and topics",
                "Conducts in-depth, multi-turn interviews with an expert AI",
                "Gathers information in parallel for efficiency",
                "Synthesizes insights into a final report",
            ],
        ),
        unsafe_allow_html=True,
    )

col31, col32, col33 = container.columns(3)

with col31:
    st.markdown(
        custom_card(
            title="Simple RAG",
            description=[],
        ),
        unsafe_allow_html=True,
    )

with col32:
    st.markdown(
        custom_card(
            title="Graph RAG",
            description=[],
        ),
        unsafe_allow_html=True,
    )
