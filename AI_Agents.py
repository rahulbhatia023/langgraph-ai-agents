import streamlit as st


def get_api_key(keys):
    for key_name, key_type in keys.items():
        if key_name not in st.session_state:
            st.session_state[key_name] = ""

        if api_key := st.text_input(
            label=f"{key_name}", value=st.session_state[key_name], type=key_type
        ):
            st.session_state[key_name] = api_key


def custom_card(title, description):
    return f"""
    <div class="custom-card">
        <h4>{title}</h4>
        {"<ul>" + "".join([f"<li>{point}</li>" for point in description]) + "</ul>"}
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
    get_api_key(
        {
            "OPENAI_API_KEY": "password",
            "E2B_API_KEY": "password",
            "FINANCIAL_DATASETS_API_KEY": "password",
            "POLYGON_API_KEY": "password",
            "TAVILY_API_KEY": "password",
            "REDDIT_CLIENT_ID": "password",
            "REDDIT_CLIENT_SECRET": "password",
            "REDDIT_USER_AGENT": "default",
        }
    )

st.markdown("<h1>Welcome to the World of AI Agents</h1>", unsafe_allow_html=True)
st.markdown("<h3>Where Intelligence Meets Innovation</h3>", unsafe_allow_html=True)
st.markdown('<div class="vertical-spacer"></div>', unsafe_allow_html=True)

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

col21, col22, _ = container.columns(3)

with col21:
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

with col22:
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
