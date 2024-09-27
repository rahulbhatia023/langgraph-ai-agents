import streamlit as st
from streamlit_card import card

st.set_page_config(
    page_title="AI Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import streamlit as st



st.markdown("""
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
    }
    
    p {
        font-size: 18px;
        line-height: 1.6;
    }
    
    .vertical-spacer {
        margin-top: 80px;
        margin-bottom: 80px;
    }

    /* Custom card styles */
    
    .custom-card {
        background-color: #1A1A1A;
        border: 1px solid #BB86FC;
        border-radius: 20px;
        padding: 20px;
        width: 400px;  /* Fixed width */
        height: 600px; /* Fixed height */
    }

    .custom-card h3 {
        color: #BB86FC;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .custom-card p {
        color: #E0E0E0;
        font-family: 'Open Sans', sans-serif;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

def custom_card(title, text, image_url):
    return f"""
    <div class="custom-card">
        <h3>{title}</h3>
        <p>{text}</p>
    </div>
    """

# Main content
st.markdown("<h1>Welcome to the World of AI Agents</h1>", unsafe_allow_html=True)
st.markdown("<h3>Discover the Power of Intelligent Automation</h3>", unsafe_allow_html=True)

# Add increased vertical space
st.markdown('<div class="vertical-spacer"></div>', unsafe_allow_html=True)

# AI Agent Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(custom_card(
        "Reactive Agents",
        "These agents react to the current state of the environment without considering past experiences.",
        "https://via.placeholder.com/150"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(custom_card(
        "Deliberative Agents",
        "These agents maintain an internal state and model of the world to make decisions.",
        "https://via.placeholder.com/150"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(custom_card(
        "Learning Agents",
        "These agents can learn from their experiences and improve their performance over time.",
        "https://via.placeholder.com/150"
    ), unsafe_allow_html=True)

with col4:
    st.markdown(custom_card(
        "Learning Agents",
        "These agents can learn from their experiences and improve their performance over time.",
        "https://via.placeholder.com/150"
    ), unsafe_allow_html=True)

