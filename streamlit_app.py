import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# app config
st.set_page_config(page_title="Rorschach Test")
st.title("Rorschach Test")


def get_response(user_query, chat_history, llm=None):
    template = """
    You are a Psychologist. You are conducting a Rorschach test.:
    You are going to first present the inkblot to the patient and then ask them what they see in the inkblot.
    The patient will then respond with what they see.
    You will then ask the patient why they see what they see.
    You will then ask the patient how they feel about what they see.
    You will then ask the patient if they have any other thoughts about the inkblot.
    You wont respond to the patient's answers. just ask the questions.
    You will then present the next inkblot to the patient and repeat the process.
    After finishing the test, you will thank the patient for their time and tell them that the test is over.
    
    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    llm = ChatOpenAI(api_key=openai_api_key)
# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Psychologist. How can I help you?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history, llm))

    st.session_state.chat_history.append(AIMessage(content=response))