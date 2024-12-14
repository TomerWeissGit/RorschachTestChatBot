import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from typing import List
from langgraph.checkpoint.memory import MemorySaver

# Initialize LangChain components
llm = ChatOpenAI(temperature=0)
search_tool = TavilySearchResults(max_results=3)

# Streamlit app config
st.set_page_config(page_title="Rorschach Test", layout="wide")
st.title("Rorschach Test")

# Define Image List
IMAGES = [f"https://www.rorschach.org/blots/rorschach-blot-{i}.jpg" for i in range(1, 11)]  # Replace with actual image URLs

# Define State Model
class RorschachTestState(BaseModel):
    current_image: int = Field(0, description="Current image index.")
    user_responses: List[dict] = Field(default_factory=list, description="List of user responses.")
    search_results: List[str] = Field(default_factory=list, description="List of search results.")
    final_report: str = Field("", description="Final report after analysis.")

# LangGraph Node Functions

# Present image and ask the first question
def present_image(state: RorschachTestState):
    if state.current_image < len(IMAGES):
        st.image(IMAGES[state.current_image], caption=f"Ink Blot {state.current_image + 1}")
        question = f"What do you see in this image (Ink Blot {state.current_image + 1})?"
        return {"messages": [HumanMessage(content=question)]}
    return {"messages": []}

# Ask the second question
def ask_location(state: RorschachTestState):
    question = f"Where exactly do you see what you described in Ink Blot {state.current_image + 1}?"
    return {"messages": [HumanMessage(content=question)]}

# Save user responses
def save_response(state: RorschachTestState, responses: List[str]):
    state.user_responses.append({
        "image": IMAGES[state.current_image],
        "response": responses
    })
    return {"user_responses": state.user_responses}

# Search the web for user responses
def search_for_response(state: RorschachTestState):
    last_response = state.user_responses[-1]
    query = f"{last_response['response'][0]} location: {last_response['response'][1]}"
    search_results = search_tool.invoke(query)
    state.search_results.append(search_results)
    return {"search_results": state.search_results}

# Move to the next image or finish the test
def next_or_finish(state: RorschachTestState):
    if state.current_image < len(IMAGES) - 1:
        state.current_image += 1
        return "present_image"
    return "generate_report"

# Generate a final report
def generate_report(state: RorschachTestState):
    context = "\n\n".join([f"User response: {resp['response']} - Search results: {sr}"
                           for resp, sr in zip(state.user_responses, state.search_results)])
    report_prompt = f"""You are a psychologist analyzing responses to a Rorschach test. 
    Based on the following data, generate a cohesive psychological report:

    {context}
    """
    report = llm.invoke([HumanMessage(content=report_prompt)])
    state.final_report = report.content
    return {"final_report": state.final_report}

# Define LangGraph
graph = StateGraph(RorschachTestState)
graph.add_node("present_image", present_image)
graph.add_node("ask_location", ask_location)
graph.add_node("save_response", save_response)
graph.add_node("search_for_response", search_for_response)
graph.add_node("next_or_finish", next_or_finish)
graph.add_node("generate_report", generate_report)

# Define edges
graph.add_edge(START, "present_image")
graph.add_edge("present_image", "ask_location")
graph.add_edge("ask_location", "save_response")
graph.add_edge("save_response", "search_for_response")
graph.add_edge("search_for_response", "next_or_finish")
graph.add_edge("next_or_finish", ["present_image", "generate_report"])
graph.add_edge("generate_report", END)

# Compile the graph
memory = MemorySaver()
compiled_graph = graph.compile(checkpointer=memory)

# Streamlit Interaction
if "state" not in st.session_state:
    st.session_state.state = RorschachTestState()

# Run Graph
state = st.session_state.state
next_node = compiled_graph.invoke(state)

# Display Final Report
if state.final_report:
    st.subheader("Final Psychological Report")
    st.write(state.final_report)
