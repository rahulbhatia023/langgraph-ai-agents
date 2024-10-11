import tempfile

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

    show_file_uploader: bool = False
    file_upload_label: str = "Upload a file"
    file_upload_type: list[str] = ["csv"]

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
                    values={"messages": [HumanMessage(content=human_message)]}
                           | cls.agent.update_graph_state(human_message),
                    as_node=cls.agent.update_as_node,
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
    def pre_render(cls):
        pass

    @classmethod
    def render(cls):
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
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<h2 class='fontStyle' style='color:#C4DAD2';>{cls.agent.name}</h2><br/>", unsafe_allow_html=True
        )

        if cls.required_keys and not keys_missing(cls.required_keys):
            if "graphs" not in st.session_state:
                st.session_state.graphs = {}
            if cls.agent.name not in st.session_state.graphs:
                st.session_state.graphs[cls.agent.name] = cls.agent.get_graph()

            agent_graph = st.session_state.graphs[cls.agent.name]

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

                if cls.show_file_uploader:
                    if "uploaded_file" not in st.session_state:
                        st.session_state.uploaded_file = {}
                    if cls.agent.name not in st.session_state.uploaded_file:
                        st.session_state.uploaded_file[cls.agent.name] = None

                    st.markdown(
                        "<br/><br/><h3 style='color:#E9EFEC;font-family: Poppins;text-align: center'>Upload SQLite DB file</h3>",
                        unsafe_allow_html=True,
                    )

                    if uploaded_file := st.file_uploader(
                            label=cls.file_upload_label,
                            type=cls.file_upload_type,
                            label_visibility="hidden",
                    ):
                        with tempfile.NamedTemporaryFile(delete=False) as file:
                            file.write(uploaded_file.read())
                            file.flush()
                            st.session_state["uploaded_file"][
                                cls.agent.name
                            ] = file.name

            if "page_messages" not in st.session_state:
                st.session_state.page_messages = {}

            if cls.agent.name not in st.session_state.page_messages:
                st.session_state.page_messages[cls.agent.name] = []

            for message in st.session_state.page_messages[cls.agent.name]:
                with st.chat_message(message["role"]):
                    st.markdown(f"<p class='fontStyle'>{message["content"]}</p>", unsafe_allow_html = True)

            if human_message := st.chat_input():
                if cls.show_file_uploader and not st.session_state.uploaded_file[cls.agent.name]:
                    st.error("Please upload a file before sending a message.", icon="🚨")
                else:
                    cls.stream_events(agent_graph=agent_graph, human_message=human_message)

    @classmethod
    def post_render(cls):
        pass

    @classmethod
    def display(cls):
        cls.pre_render()
        cls.render()
        cls.post_render()
