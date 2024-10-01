import os

import streamlit as st


def get_api_key(key_name):
    if key_name not in st.session_state:
        st.session_state[key_name] = os.getenv(key_name, "")

    api_key = st.text_input(
        label=f"{key_name}",
        type="password",
        value=st.session_state[key_name],
        key=f"{key_name}_input",
    )

    if api_key:
        st.session_state[key_name] = api_key
        os.environ[key_name] = api_key


def custom_card(title, description):
    return f"""
    <div class="custom-card">
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """


st.set_page_config(page_title="AI Agents", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .stApp {
        background-color: #16423C;
        font-family: 'Poppins';
    }

    h1 {
        font-family: 'Poppins';
        color: #C4DAD2;
        text-align: center;
    }

    h3 {
        font-family: 'Poppins';
        color: #E9EFEC;
        text-align: center;
        font-size: 20px;
        font-weight: 10;
    }

    p {
        font-family: 'Poppins';
        color: #C4DAD2;
    }
    
    .vertical-spacer {
        margin-top: 80px;
        margin-bottom: 80px;
    }

    .custom-card {
        background-color: #6A9C89;
        border: 2px solid #FFFFFF;
        border-radius: 20px;
        padding: 20px;
        height: 405px;
        margin-bottom: 30px;
        margin-left: 10px;
    }

    .custom-card h4 {
        color: #16423C;
        font-family: 'Poppins';
        font-weight: 600;
        margin-bottom: 10px;
        text-align: center;
    }

    .custom-card p {
        color: #FFFFFF;
        font-family: 'Poppins';
        font-size: 15px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    get_api_key("OPENAI_API_KEY")
    get_api_key("E2B_API_KEY")
    get_api_key("FINANCIAL_DATASETS_API_KEY")
    get_api_key("POLYGON_API_KEY")

st.markdown("<h1>Welcome to the World of AI Agents</h1>", unsafe_allow_html=True)
st.markdown("<h3>Where Intelligence Meets Innovation</h3>", unsafe_allow_html=True)
st.markdown('<div class="vertical-spacer"></div>', unsafe_allow_html=True)

container = st.container()

col11, col12, col13 = container.columns(3)

with col11:
    st.markdown(
        custom_card(
            title="Code Assistant",
            description="AI-powered assistant that integrates Python execution capabilities with React component rendering on the fly, offering a comprehensive environment for data analysis, visualization, and interactive web development.",
        ),
        unsafe_allow_html=True,
    )

with col12:
    st.markdown(
        custom_card(
            "Financial Advisor",
            "This financial agent integrates a stock market API to provide real-time stock data and detailed financial insights. Additionally, it incorporates the Tavily search API for broader web searches, offering a comprehensive tool for financial data and information retrieval.",
        ),
        unsafe_allow_html=True,
    )

with col13:
    st.markdown(
        custom_card(
            "Reddit Search",
            "This AI agent searches Reddit to find the most relevant answers to user queries across multiple subreddits. It intelligently analyzes responses, filters out the noise, and delivers the best possible answer by consolidating insights from various Reddit communities, providing users with accurate and curated information.",
        ),
        unsafe_allow_html=True,
    )

col21, col22, _ = container.columns(3)

with col21:
    st.markdown(
        custom_card(
            title="Research Analyst",
            description="This agent customizes the research process by assembling AI analysts based on user-selected sources and topics. It conducts in-depth, multi-turn interviews with an expert AI to extract detailed insights, gathers information in parallel, and synthesizes the findings into a final report",
        ),
        unsafe_allow_html=True,
    )

with col22:
    st.markdown(
        custom_card(
            "SQL Generator",
            "This agent bridges the gap between natural language questions and dataset, allowing users to question about a dataset and receive insightful response. Users can ask questions about the data in natural language. The agent generates a SQL query based on the user's question, executes it on the dataset, and then displays the result and generate the human language response.",
        ),
        unsafe_allow_html=True,
    )
