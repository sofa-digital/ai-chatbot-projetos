import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from classes import FinalResponse
import sofia_logic
import admin_ui

st.set_page_config(
    page_title="SOFIA Chat - Projetos",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={}
)

def initialize_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "show_login" not in st.session_state:
        st.session_state.show_login = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "needs_restart" not in st.session_state:
        st.session_state.needs_restart = False
    if "repository" not in st.session_state:
        st.session_state.repository = None
    if "graph" not in st.session_state:
        st.session_state.graph = sofia_logic.create_graph()

initialize_session_state()

def restart_application():
    history_backup = None
    if "history" in st.session_state:
        history_backup = st.session_state.history
    for key in list(st.session_state.keys()):
        if key != "authenticated" and key != "show_login":
            del st.session_state[key]
    if history_backup is not None:
        st.session_state.history = history_backup
    sofia_logic.cleanup_memory()
    st.session_state.needs_restart = True
    initialize_session_state()
    st.rerun()

if st.session_state.needs_restart:
    st.session_state.needs_restart = False
    st.success("Application restarted successfully!")

BUCKET_NAME = "docs-projetos-chatbot"

try:
    aws_config = sofia_logic.configure_aws()
except Exception as e:
    if st.session_state.authenticated:
        st.sidebar.error(f"Error configuring AWS: {str(e)}")

try:
    if "repository" not in st.session_state or st.session_state.repository is None:
        repository, vectorstore = sofia_logic.get_repository(BUCKET_NAME)
        st.session_state.repository = repository
    else:
        repository = st.session_state.repository
        if not repository._vectorstore:
            _, vectorstore = sofia_logic.get_repository(BUCKET_NAME)
except Exception as e:
    if st.session_state.authenticated:
        st.sidebar.error(f"Error loading repository: {str(e)}")
    repository = None

st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
    <img src="https://sofatima.s3.sa-east-1.amazonaws.com/sofia.png" alt="SOFIA Logo" style="height:60px;">
    <h1 style="margin: 0;">SOFIA Projetos: v25.4.29.1013</h1>
</div>
""", unsafe_allow_html=True)


if st.session_state.show_login and not admin_ui.check_password():
    pass
elif st.session_state.authenticated:
    admin_ui.render_admin_sidebar(BUCKET_NAME)

if not st.session_state.show_login or st.session_state.authenticated:
    for message in st.session_state.history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    query = st.chat_input("Ask SOFIA something...")
    if query =="admin":
        st.session_state.show_login = True
        st.rerun()
    if query:
        st.session_state.history.append(HumanMessage(content=query))
        
        with st.chat_message("user"):
            st.markdown(query)
            
        context = ""
        for i, msg in enumerate(st.session_state.history[:-1]):
            if isinstance(msg, HumanMessage):
                context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"Assistant: {msg.content}\n"
                
        MAX_HISTORY = 10
        if len(st.session_state.history) > MAX_HISTORY:
            st.session_state.history = st.session_state.history[-MAX_HISTORY:]
            
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.graph.invoke(st.session_state.history)
                    final_result_json = response[-1].content
                    final_result_pydantic = FinalResponse.model_validate_json(final_result_json)
                    answer = final_result_pydantic.answer
                    st.session_state.history.append(AIMessage(content=answer))
                    st.markdown(answer)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.session_state.history.append(AIMessage(content=error_message))
            with st.chat_message("assistant"):
                st.markdown(error_message)
    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    