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
        border: 1px solid #BB86FC;
        border-radius: 20px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .custom-card h4 {
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

    .custom-card img {
        width: 100%;
        height: 150px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def custom_card(title, description, image_url):
    return f"""
    <div class="custom-card">
        <img src="{image_url}" alt="{title}">
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """


# Main content
st.markdown("<h1>Welcome to the World of AI Agents</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3>Discover the Power of Intelligent Automation</h3>", unsafe_allow_html=True
)

# Add increased vertical space
st.markdown('<div class="vertical-spacer"></div>', unsafe_allow_html=True)

# AI Agent Cards
# Use st.container() to create a flexible container
container = st.container()

# Use columns with equal width inside the container
col1, col2, col3 = container.columns(3)

with col1:
    st.markdown(
        custom_card(
            "Reactive Agents",
            "These agents react to the current state of the environment without considering past experiences.",
            "https://via.placeholder.com/150"
        ),
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        custom_card(
            "Deliberative Agents",
            "These agents maintain an internal state and model of the world to make decisions.",
            "https://via.placeholder.com/150"
        ),
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        custom_card(
            "Learning Agents",
            "These agents can learn from their experiences and improve their performance over time.",
            "https://via.placeholder.com/150"
        ),
        unsafe_allow_html=True
    )
