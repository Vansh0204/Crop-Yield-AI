import joblib
import json
import os
import difflib
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

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
        new_area = matches[0] if matches else "India"
        warnings.append(f"Area '{area}' not recognized. Using closest match: '{new_area}'.")
        area = new_area
    
    crop = state.get("crop", "Wheat")
    if crop not in known["crops"]:
        warnings.append(f"Crop '{crop}' not recognized. Defaulting to 'Wheat'.")
        crop = "Wheat"

    rainfall = max(0.0, min(5000.0, float(state.get("rainfall", 1050.0))))
    temp = max(-10.0, min(55.0, float(state.get("temperature", 20.0))))
    pest = max(0.0, min(500000.0, float(state.get("pesticides", 12000))))
    year = max(1990, min(2030, int(state.get("year", 2024))))

    return {
        "area": area, "crop": crop, "rainfall": rainfall, 
        "temperature": temp, "pesticides": pest, "year": year,
        "warnings": warnings
    }

def predict_yield_node(state: FarmAdvisoryState):
    return {"predicted_yield": 42000.0, "yield_category": "Medium"}

def run_advisory(inputs):
    workflow = StateGraph(FarmAdvisoryState)
    workflow.add_node("validate", validate_inputs_node)
    workflow.add_node("predict", predict_yield_node)
    workflow.set_entry_point("validate")
    workflow.add_edge("validate", "predict")
    return workflow.compile().invoke(inputs)
