import streamlit as st


def add_chat_message(agent_name: str, role: str, content: str):
    st.session_state.page_messages[agent_name].append(
        {"role": role, "content": content}
    )
    with st.chat_message(role):
        st.markdown(content)
