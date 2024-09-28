import streamlit as st

st.set_page_config(page_title="AI Agents", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Open Sans:wght@300;400;600;700&display=swap');
    
    .stApp {
        background-color: #2A2A2A;
    }

    h1 {
        font-family: 'Poppins', sans-serif;
        color: #BB86FC;
        text-align: center;
    }

    h3 {
        font-family: 'Poppins', sans-serif;
        color: #BB86FA;
        text-align: center;
        font-size: 20px;
        font-weight: 10;
    }

    p {
        font-size: 18px;
        line-height: 1.6;
        color: #E0E0E0;
    }

    .vertical-spacer {
        margin-top: 80px;
        margin-bottom: 80px;
    }

    /* Custom card styles */
    .custom-card {
        background-color: #1A1A1A;
        border: 1px solid #FFFFFF;
        border-radius: 20px;
        padding: 20px;
        height: 405px;
        margin-bottom: 30px;
        margin-left: 10px;
        
    }

    .custom-card h4 {
        color: #BB86FC;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        margin-bottom: 10px;
        text-align: center;
    }

    .custom-card p {
        font-family: 'Poppins', sans-serif;
        font-size: 15px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def custom_card(title, description):
    return f"""
    <div class="custom-card">
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """


# Main content
st.markdown("<h1>Welcome to the World of AI Agents</h1>", unsafe_allow_html=True)
st.markdown("<h3>Where Intelligence Meets Innovation</h3>", unsafe_allow_html=True)

# Add increased vertical space
st.markdown('<div class="vertical-spacer"></div>', unsafe_allow_html=True)

# AI Agent Cards
# Use st.container() to create a flexible container
container = st.container()

# Use columns with equal width inside the container
col11, col12, col13 = container.columns(3)

with col11:
    st.markdown(
        custom_card(
            title="Code Assistant",
            description="AI-powered assistant that integrates Python execution capabilities with React component rendering on the fly, offering a comprehensive environment for data analysis, visualization, and interactive web development."
        ),
        unsafe_allow_html=True,
    )

with col12:
    st.markdown(
        custom_card(
            "Financial Advisor",
            "This financial agent integrates a stock market API to provide real-time stock data and detailed financial insights. Additionally, it incorporates the Tavily search API for broader web searches, offering a comprehensive tool for financial data and information retrieval."
        ),
        unsafe_allow_html=True,
    )

with col13:
    st.markdown(
        custom_card(
            "Reddit Search",
            "This AI agent searches Reddit to find the most relevant answers to user queries across multiple subreddits. It intelligently analyzes responses, filters out the noise, and delivers the best possible answer by consolidating insights from various Reddit communities, providing users with accurate and curated information."
        ),
        unsafe_allow_html=True,
    )

st.markdown('<div class="row-spacer"></div>', unsafe_allow_html=True)

col21, col22, _ = container.columns(3)

with col21:
    st.markdown(
        custom_card(
            title="Research Analyst",
            description="This agent customizes the research process by assembling AI analysts based on user-selected sources and topics. It conducts in-depth, multi-turn interviews with an expert AI to extract detailed insights, gathers information in parallel, and synthesizes the findings into a final report"
        ),
        unsafe_allow_html=True,
    )

with col22:
    st.markdown(
        custom_card(
            "SQL Generator",
            "This agent bridges the gap between natural language questions and dataset, allowing users to question about a dataset and receive insightful response. Users can ask questions about the data in natural language. The agent generates a SQL query based on the user's question, executes it on the dataset, and then displays the result and generate the human language response."
        ),
        unsafe_allow_html=True,
    )
