import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from streamlit.commands.page_config import Layout, PageIcon

from common.agent import BaseAgent
from common.chat import add_chat_message, display_message


def keys_missing(required_keys: list[str]):
    if_missing = False
    for key in required_keys:
        if key not in st.session_state or not st.session_state[key]:
            st.error(f"Please enter {key} in the home page.", icon="🚨")
            if_missing = True
    return if_missing


class BasePage:
    agent: BaseAgent = None
    required_keys: list[str] = []
    page_icon: PageIcon = "🤖"
    layout: Layout = "wide"

    @classmethod
    def pre_render(cls):
        pass

    @classmethod
    def stream_events(cls, agent_graph, human_message):
        config = {"configurable": {"thread_id": "1"}}

        def is_first_human_message():
            for message in st.session_state.page_messages[cls.agent.name]:
                if message.get("role") == "human":
                    return False
            return True

        if human_message:
            if is_first_human_message():
                agent_input = {
                    "messages": [
                        SystemMessage(content=cls.agent.system_prompt),
                        HumanMessage(content=human_message),
                    ]
                }
            elif not cls.agent.interrupt_before:
                agent_input = {
                    "messages": [
                        HumanMessage(content=human_message),
                    ]
                }
            else:
                agent_input = None
                agent_graph.update_state(
                    config=config,
                    values={"messages": [HumanMessage(content=human_message)]},
                    as_node="agent",
                )

            add_chat_message(
                agent_name=cls.agent.name, role="human", content=human_message
            )

            for event in agent_graph.stream(
                input=agent_input,
                config=config,
                stream_mode="updates",
            ):
                for k, v in event.items():
                    if cls.agent.nodes_to_display:
                        if k in cls.agent.nodes_to_display:
                            display_message(agent_name=cls.agent.name, v=v)
                    else:
                        display_message(agent_name=cls.agent.name, v=v)

    @classmethod
    def render(cls):
        if "graphs" not in st.session_state:
            st.session_state.graphs = {}
        if cls.agent.name not in st.session_state.graphs:
            st.session_state.graphs[cls.agent.name] = cls.agent.get_graph()

        if cls.required_keys and not keys_missing(cls.required_keys):
            agent_graph = st.session_state.graphs[cls.agent.name]

            st.set_page_config(
                page_title=cls.agent.name, page_icon=cls.page_icon, layout=cls.layout
            )

            st.markdown(
                """
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

                    .stApp {
                        font-family: 'Poppins';
                        background-color: #16423C;
                    }

                    .fontStyle {
                        font-family: 'Poppins';
                        color: #C4DAD2;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<h2 class='fontStyle'>{cls.agent.name}</h2>", unsafe_allow_html=True
            )

            with st.sidebar:
                st.markdown(
                    "<h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>LangGraph Workflow Visualization</h3>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <style>
                        [data-testid="stImage"] {
                            border-radius: 10px;
                            overflow: hidden;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.image(
                    agent_graph.get_graph().draw_mermaid_png(),
                    use_column_width="always",
                )

            if "page_messages" not in st.session_state:
                st.session_state.page_messages = {}

            if cls.agent.name not in st.session_state.page_messages:
                st.session_state.page_messages[cls.agent.name] = []

            for message in st.session_state.page_messages[cls.agent.name]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            cls.stream_events(agent_graph=agent_graph, human_message=st.chat_input())

    @classmethod
    def post_render(cls):
        pass

    @classmethod
    def display(cls):
        cls.pre_render()
        cls.render()
        cls.post_render()
