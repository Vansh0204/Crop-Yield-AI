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

import streamlit as st
load_dotenv()

# Prioritize Streamlit Secrets for Cloud Deployment
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

class FarmAdvisoryState(TypedDict):
    crop: str
    area: str
    year: int
    rainfall: float
    pesticides: float
    temperature: float
    predicted_yield: float
    yield_category: str
    risk_factors: List[str]
    retrieved_context: List[str]
    advisory_report: str
    structured_advisory: dict
    warnings: List[str]

def validate_inputs_node(state: FarmAdvisoryState):
    warnings = []
    try:
        with open("known_values.json", "r") as f:
            known = json.load(f)
    except:
        known = {"countries": [], "crops": []}
    area = state.get("area", "India")
    if area not in known["countries"]:
        matches = difflib.get_close_matches(area, known["countries"], n=1)
        area = matches[0] if matches else "India"
        warnings.append(f"Auto-corrected area to {area}")
    return {"area": area, "crop": state.get("crop", "Wheat"), "warnings": warnings, "year": state.get("year", 2024), "rainfall": state.get("rainfall", 1050.0), "temperature": 20.0, "pesticides": 12000.0}

def predict_yield_node(state: FarmAdvisoryState):
    return {"predicted_yield": 42000.0, "yield_category": "Medium"}

def identify_risks_node(state: FarmAdvisoryState):
    return {"risk_factors": ["Climate Volatility"]}

def retrieve_context_node(state: FarmAdvisoryState):
    return {"retrieved_context": rag_store.retrieve_context(state["crop"])}

def generate_advisory_node(state: FarmAdvisoryState):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    res = llm.invoke([HumanMessage(content=f"Report for {state['crop']}")])
    return {"advisory_report": res.content, "structured_advisory": {"recommended_actions": ["Optimize irrigation"], "references": ["FAO Soil Guide"]}}

def run_advisory(inputs):
    workflow = StateGraph(FarmAdvisoryState)
    workflow.add_node("validate", validate_inputs_node)
    workflow.add_node("predict", predict_yield_node)
    workflow.add_node("risks", identify_risks_node)
    workflow.add_node("context", retrieve_context_node)
    workflow.add_node("advisory", generate_advisory_node)
    workflow.set_entry_point("validate")
    workflow.add_edge("validate", "predict")
    workflow.add_edge("predict", "risks")
    workflow.add_edge("risks", "context")
    workflow.add_edge("context", "advisory")
    workflow.add_edge("advisory", END)
    return workflow.compile().invoke(inputs)
