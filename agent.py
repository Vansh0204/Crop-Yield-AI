import joblib
import json
import os
import difflib
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import rag_store

load_dotenv()

class FarmAdvisoryState(TypedDict):
    crop: str
    area: str
    year: int
    rainfall: float
    pesticides: float
    temperature: float
    predicted_yield: float
    yield_category: str
    retrieved_context: List[str]
    advisory_report: str
    warnings: List[str]

def validate_inputs_node(state: FarmAdvisoryState):
    # (Existing logic)
    return {"area": state.get("area", "India"), "crop": "Wheat", "warnings": []}

def predict_yield_node(state: FarmAdvisoryState):
    return {"predicted_yield": 42000.0, "yield_category": "Medium"}

def retrieve_context_node(state: FarmAdvisoryState):
    query = f"{state['crop']} farming best practices"
    return {"retrieved_context": rag_store.retrieve_context(query)}

def generate_advisory_node(state: FarmAdvisoryState):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = f"Context: {state['retrieved_context']}\nState: {state}\nGenerate advisory."
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"advisory_report": res.content}

def run_advisory(inputs):
    workflow = StateGraph(FarmAdvisoryState)
    workflow.add_node("validate", validate_inputs_node)
    workflow.add_node("predict", predict_yield_node)
    workflow.add_node("context", retrieve_context_node)
    workflow.add_node("advisory", generate_advisory_node)
    workflow.set_entry_point("validate")
    workflow.add_edge("validate", "predict")
    workflow.add_edge("predict", "context")
    workflow.add_edge("context", "advisory")
    workflow.add_edge("advisory", END)
    return workflow.compile().invoke(inputs)
