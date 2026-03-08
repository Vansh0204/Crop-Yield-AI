"""
LangGraph agentic workflow for crop yield prediction and farm advisory.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, TypedDict

import joblib
import pandas as pd
from langgraph.graph import END, START, StateGraph

from llm_client import get_llm_response

# --- Reference stats (from training data) ------------------------------------

_DATA_PATH = "data/crop_yield.csv"


@lru_cache(maxsize=1)
def _yield_percentile_thresholds() -> tuple[float, float]:
    df = pd.read_csv(_DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    series = df["hg/ha_yield"].dropna()
    return float(series.quantile(0.33)), float(series.quantile(0.67))


@lru_cache(maxsize=1)
def _year_bounds() -> tuple[int, int]:
    df = pd.read_csv(_DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    years = df["Year"].dropna().astype(int)
    return int(years.min()), int(years.max())


# --- Agronomic context (10 crops from dataset) -------------------------------

CROP_BEST_PRACTICES: dict[str, str] = {
    "Cassava": (
        "Cassava: use disease-free stakes, wide spacing on well-drained soils, "
        "weed control in the first 8–12 weeks, and harvest when roots mature; "
        "avoid waterlogging and rotate to break pest cycles."
    ),
    "Maize": (
        "Maize: match planting to soil temperature, maintain adequate N through "
        "split applications where feasible, control fall armyworm early, and "
        "ensure even plant density for pollination."
    ),
    "Plantains and others": (
        "Plantains: maintain steady soil moisture, mulch to reduce evaporation, "
        "remove suckers to focus the mat, and protect bunches from wind/sun scald."
    ),
    "Potatoes": (
        "Potatoes: use certified seed tubers, hilling for tuber development, "
        "blight scouting with timely fungicides, and stop irrigation before senescence."
    ),
    "Rice, paddy": (
        "Rice (paddy): level fields for uniform flooding, manage nursery quality, "
        "optimize transplanting density, and monitor water depth to reduce weeds and pests."
    ),
    "Sorghum": (
        "Sorghum: plant into warm soil, manage Striga where present, "
        "use drought-tolerant varieties in marginal rainfall, and bird-proof heads near maturity."
    ),
    "Soybeans": (
        "Soybeans: inoculate with rhizobia where adapted, avoid late planting that "
        "shortens filling, scout for pod feeders, and harvest at correct moisture to reduce shatter."
    ),
    "Sweet potatoes": (
        "Sweet potatoes: use clean vines, loose soils for root expansion, "
        "weed early until vines cover rows, and cure roots properly after harvest."
    ),
    "Wheat": (
        "Wheat: time sowing to avoid frost and heat stress at grain fill, "
        "manage foliar diseases with rotation and timely sprays, and optimize seeding rate to tiller well."
    ),
    "Yams": (
        "Yams: plant quality seed yams on mounds/ridges for drainage, "
        "trellis or stake vines where needed, and store carefully to reduce rot."
    ),
}

_DEFAULT_CROP_CONTEXT = (
    "General: verify seed quality, soil pH and organic matter, balanced nutrition, "
    "integrated pest management, and local extension guidance for the crop and region."
)


def _agronomic_context_for_crop(crop: str) -> str:
    return CROP_BEST_PRACTICES.get(crop, _DEFAULT_CROP_CONTEXT)


# --- State -------------------------------------------------------------------


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
    recommendations: List[str]
    advisory_report: str
    status: str


# --- Nodes -------------------------------------------------------------------


def predict_yield_node(state: FarmAdvisoryState) -> dict:
    model = joblib.load("model/crop_model.pkl")
    input_data = pd.DataFrame(
        {
            "Area": [state["area"]],
            "Item": [state["crop"]],
            "Year": [state["year"]],
            "average_rain_fall_mm_per_year": [state["rainfall"]],
            "pesticides_tonnes": [state["pesticides"]],
            "avg_temp": [state["temperature"]],
        }
    )
    input_encoded = pd.get_dummies(input_data)
    model_features = list(model.feature_names_in_)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
    predicted = float(model.predict(input_encoded)[0])

    p33, p67 = _yield_percentile_thresholds()
    if predicted < p33:
        category = "Low"
    elif predicted > p67:
        category = "High"
    else:
        category = "Medium"

    return {
        "predicted_yield": predicted,
        "yield_category": category,
        "status": "predicted",
    }


def identify_risks_node(state: FarmAdvisoryState) -> dict:
    risks: List[str] = []
    y_min, y_max = _year_bounds()

    if state["rainfall"] < 600:
        risks.append("Rainfall is below 600 mm/year — elevated drought stress risk.")
    if state["temperature"] > 35:
        risks.append("Average temperature above 35 °C — heat stress risk for many crops.")
    if state["temperature"] < 10:
        risks.append("Average temperature below 10 °C — cold stress / slow development risk.")
    if state["pesticides"] > 50000:
        risks.append("Pesticide use above 50,000 tonnes (input scale) — review application rates and stewardship.")
    if state["year"] < y_min or state["year"] > y_max:
        risks.append(
            f"Year {state['year']} is outside the training year range ({y_min}–{y_max}) — extrapolation uncertainty."
        )

    return {"risk_factors": risks, "status": "risks_identified"}


def retrieve_context_node(state: FarmAdvisoryState) -> dict:
    ctx = _agronomic_context_for_crop(state["crop"])
    # Staging for generate_advisory_node prompt; overwritten after LLM call.
    return {"advisory_report": ctx, "status": "context_retrieved"}


def generate_advisory_node(state: FarmAdvisoryState) -> dict:
    agronomic_context = state["advisory_report"] or _agronomic_context_for_crop(state["crop"])

    prompt = f"""You are an agronomy advisor. Using the structured farm state below, produce a concise, practical advisory.

Farm state:
- Area (country): {state["area"]}
- Crop: {state["crop"]}
- Year: {state["year"]}
- Rainfall (mm/year): {state["rainfall"]}
- Pesticides (tonnes): {state["pesticides"]}
- Average temperature (°C): {state["temperature"]}
- Model predicted yield (hg/ha): {state["predicted_yield"]:.2f}
- Yield category (vs historical dataset tertiles): {state["yield_category"]}
- Risk flags: {state["risk_factors"]}
- Reference best practices for this crop: {agronomic_context}

Respond in this exact structure (plain text, no JSON):
SUMMARY: (2–4 sentences)
RECOMMENDATIONS:
- (bullet 1)
- (bullet 2)
- (bullet 3)
- (optional additional bullets)
RISKS_TO_MONITOR: (short paragraph referencing the risk flags where relevant)
"""

    llm_text = get_llm_response(prompt)
    recommendations = _parse_recommendations_from_llm(llm_text)

    return {
        "recommendations": recommendations,
        "advisory_report": llm_text,
        "status": "advisory_generated",
    }


def _parse_recommendations_from_llm(llm_text: str) -> List[str]:
    """Extract bullet lines under RECOMMENDATIONS: if present; else heuristic split."""
    lines = llm_text.splitlines()
    recs: List[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("RECOMMENDATIONS:"):
            in_block = True
            continue
        if in_block:
            if stripped.upper().startswith(("SUMMARY:", "RISKS_TO_MONITOR:", "---")):
                break
            if stripped.startswith(("-", "*", "•")):
                recs.append(stripped.lstrip("-*• ").strip())
            elif stripped == "" and recs:
                # allow blank lines inside block
                continue
            elif stripped and not stripped.startswith("-"):
                # next section without header
                if stripped.upper().startswith("RISKS"):
                    break
                recs.append(stripped)
    if recs:
        return recs
    # Fallback: non-empty lines that look like advice
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().upper().startswith("SUMMARY:")][:5]


def format_report_node(state: FarmAdvisoryState) -> dict:
    ctx = _agronomic_context_for_crop(state["crop"])
    y_min, y_max = _year_bounds()
    p33, p67 = _yield_percentile_thresholds()

    risks_md = "\n".join(f"- {r}" for r in state["risk_factors"]) or "- None flagged by rule checks."
    recs_md = "\n".join(f"- {r}" for r in state["recommendations"]) or "- See AI advisory section below."

    report = f"""# Farm advisory report

## Summary
- **Area:** {state["area"]}
- **Crop:** {state["crop"]}
- **Year:** {state["year"]} (training data years: {y_min}–{y_max})
- **Predicted yield:** {state["predicted_yield"]:,.2f} hg/ha
- **Yield category:** {state["yield_category"]} (vs dataset 33rd/67th percentiles: {p33:,.0f} / {p67:,.0f} hg/ha)

## Inputs
| Field | Value |
|------|------|
| Rainfall (mm/year) | {state["rainfall"]} |
| Pesticides (tonnes) | {state["pesticides"]} |
| Avg temperature (°C) | {state["temperature"]} |

## Risk factors
{risks_md}

## Agronomic context (best practices)
{ctx}

## AI-generated advisory
{state["advisory_report"]}

## Parsed recommendations
{recs_md}
"""
    return {"advisory_report": report, "status": "report_formatted"}


# --- Graph -------------------------------------------------------------------


def _build_advisory_graph():
    graph = StateGraph(FarmAdvisoryState)
    graph.add_node("predict_yield", predict_yield_node)
    graph.add_node("identify_risks", identify_risks_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_advisory", generate_advisory_node)
    graph.add_node("format_report", format_report_node)

    graph.add_edge(START, "predict_yield")
    graph.add_edge("predict_yield", "identify_risks")
    graph.add_edge("identify_risks", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_advisory")
    graph.add_edge("generate_advisory", "format_report")
    graph.add_edge("format_report", END)
    return graph.compile()


_ADVISORY_APP = _build_advisory_graph()


def run_advisory(inputs: dict) -> FarmAdvisoryState:
    """
    Run the linear advisory workflow.

    Expected keys in ``inputs``: crop, area, year, rainfall, pesticides, temperature.
    """
    state: FarmAdvisoryState = {
        "crop": str(inputs["crop"]),
        "area": str(inputs["area"]),
        "year": int(inputs["year"]),
        "rainfall": float(inputs["rainfall"]),
        "pesticides": float(inputs["pesticides"]),
        "temperature": float(inputs["temperature"]),
        "predicted_yield": 0.0,
        "yield_category": "",
        "risk_factors": [],
        "recommendations": [],
        "advisory_report": "",
        "status": "initialized",
    }
    result = _ADVISORY_APP.invoke(state)
    return result  # type: ignore[return-value]


__all__ = [
    "FarmAdvisoryState",
    "run_advisory",
    "predict_yield_node",
    "identify_risks_node",
    "retrieve_context_node",
    "generate_advisory_node",
    "format_report_node",
]
