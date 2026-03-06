"""
agent.py — Context-Aware AI Chatbot Agent for the Vayu-Rakshak System.

Architecture:
  - LangChain SQL Agent connected to the SQLite database
  - Custom tool: `get_nearby_pois(lat, lon)` — queries the OpenStreetMap Overpass API
    to find factories, schools, hospitals, petrol stations, and heavy traffic roads
    within 1 km of a sensor. This lets the agent understand WHY a reading is elevated.
  - System prompt: The agent acts as a senior environmental scientist with full
    knowledge of the sensor network and the ability to correlate spatial context
    (nearby POIs) with pollution patterns.

Usage (from app.py):
    from agent import get_agent_executor
    executor = get_agent_executor()
    response = executor.invoke({"input": "Why is PM2.5 high at ARI-1885 in the morning?"})
    answer = response["output"]
"""

import os
import json
import logging
import requests
from typing import Optional

from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Database connection (same SQLite file)
# ─────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "air_quality.db")
_DB_URI  = f"sqlite:///{_DB_PATH}"


# ─────────────────────────────────────────────
# Custom Tool: Nearby Points of Interest
# ─────────────────────────────────────────────

@tool
def get_nearby_pois(lat_lon: str) -> str:
    """
    Fetch Points of Interest (POIs) near a sensor location using the
    OpenStreetMap Overpass API.

    Input:
        lat_lon (str): Comma-separated latitude and longitude, e.g. "28.67, 77.22"

    Returns:
        A human-readable summary of nearby environmental factors — including
        factories/industrial sites, schools, hospitals, petrol stations, and
        major roads — within a 1 km radius. Use this to explain why pollution
        levels may be elevated at a particular sensor location.

    Examples:
        get_nearby_pois("28.6667, 77.2283")
    """
    try:
        parts  = [p.strip() for p in lat_lon.split(",")]
        lat, lon = float(parts[0]), float(parts[1])
    except Exception:
        return (
            "Could not parse coordinates. Please provide them as 'lat, lon', "
            "e.g. '28.67, 77.22'."
        )

    radius = 1000  # metres

    # Overpass QL — fetches several categories of POI in one query
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["landuse"="industrial"](around:{radius},{lat},{lon});
      way["landuse"="industrial"](around:{radius},{lat},{lon});
      node["amenity"="school"](around:{radius},{lat},{lon});
      node["amenity"="university"](around:{radius},{lat},{lon});
      node["amenity"="college"](around:{radius},{lat},{lon});
      node["amenity"="hospital"](around:{radius},{lat},{lon});
      node["amenity"="fuel"](around:{radius},{lat},{lon});
      way["highway"~"^(primary|secondary|trunk|motorway)$"](around:{radius},{lat},{lon});
      node["man_made"="chimney"](around:{radius},{lat},{lon});
      node["man_made"="works"](around:{radius},{lat},{lon});
      node["industrial"="factory"](around:{radius},{lat},{lon});
    );
    out body;
    """

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=overpass_query,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        return "Overpass API timed out. Please try again."
    except Exception as e:
        return f"Failed to contact Overpass API: {e}"

    elements = data.get("elements", [])
    if not elements:
        return (
            f"No significant pollution sources found within {radius} m of "
            f"({lat:.4f}, {lon:.4f}). The elevation may be due to regional "
            "atmospheric conditions or long-range transport."
        )

    # Categorise results
    categories: dict[str, list[str]] = {
        "🏭 Industrial / Factory":  [],
        "🏫 School / University":   [],
        "🏥 Hospital":              [],
        "⛽ Petrol Station":        [],
        "🛣️ Major Road":            [],
        "🏗️ Other":                 [],
    }

    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", tags.get("ref", "Unnamed"))
        landuse   = tags.get("landuse", "")
        amenity   = tags.get("amenity", "")
        highway   = tags.get("highway", "")
        man_made  = tags.get("man_made", "")
        industrial= tags.get("industrial", "")

        if landuse == "industrial" or man_made in ("chimney", "works") or industrial == "factory":
            categories["🏭 Industrial / Factory"].append(name)
        elif amenity in ("school", "university", "college"):
            categories["🏫 School / University"].append(name)
        elif amenity == "hospital":
            categories["🏥 Hospital"].append(name)
        elif amenity == "fuel":
            categories["⛽ Petrol Station"].append(name)
        elif highway in ("primary", "secondary", "trunk", "motorway"):
            categories["🛣️ Major Road"].append(name)
        else:
            categories["🏗️ Other"].append(name)

    lines = [f"📍 POIs within {radius} m of ({lat:.4f}, {lon:.4f}):\n"]
    for cat, items in categories.items():
        if items:
            unique = list(dict.fromkeys(items))[:8]  # deduplicate, cap at 8
            lines.append(f"  {cat}: {', '.join(unique)}")

    if not any(categories.values()):
        lines.append("  No categorised sources found.")

    # Provide an interpretive hint
    lines.append(
        "\n💡 Interpretation: Industrial sites and major roads are primary sources "
        "of PM2.5. Schools and hospitals indicate sensitive populations. Morning "
        "traffic peaks on nearby roads often cause PM2.5 spikes between 07:00–10:00."
    )

    return "\n".join(lines)


@tool
def control_app_ui(action: str, parameters_json: str) -> str:
    """
    Control the application's user interface (navigation, map zooming, etc.).
    
    Input:
        action (str): The UI action to perform. One of:
            - "navigate_to": Switch to a different tab.
            - "zoom_to": Center and zoom the map on specific coordinates.
        parameters_json (str): A JSON string of parameters for the action.
            - For "navigate_to": {"tab": "🗺️ Dashboard" | "🚨 Surveillance" | "➕ Register Sensor" | "📈 Historical Analytics", "sensor_id": "optional_id"}
            - For "zoom_to": {"lat": float, "lon": float, "zoom": int}

    Returns:
        A confirmation message that the UI command was sent. This tool doesn't 
        actually change the UI itself; it signals the frontend to do so.
    """
    try:
        params = json.loads(parameters_json)
        return f"UI_SIGNAL: action={action}, params={json.dumps(params)}"
    except Exception as e:
        return f"Error parsing UI parameters: {e}"


# ─────────────────────────────────────────────
# Agent Factory
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """You are Dr. Vayu, a senior environmental scientist and air quality analyst 
specialising in urban pollution monitoring. You have direct access to a real-time database of 
IoT sensor readings across the city and a tool to inspect the physical environment near each sensor.

Your capabilities:
1. Query the SQLite database (tables: sensor_registry, sensor_readings) to retrieve sensor data.
2. Use the get_nearby_pois tool to identify nearby pollution sources — factories, schools, major 
   roads, petrol stations — that could explain elevated readings.
3. Use the control_app_ui tool to navigate the application or adjust the map view for the user.
4. Synthesise database evidence with spatial context to give scientifically rigorous, 
   actionable answers.

Database schema:
- sensor_registry(sensor_id, location_name, lat, long, installation_date, api_key)
- sensor_readings(id, sensor_id, timestamp, temperature, humidity, pm2p5_raw, pm2p5_corrected, 
                  is_anomaly, is_failure)

Behaviour guidelines:
- ALWAYS query the database for sensor coordinates before calling get_nearby_pois.
- When asked about elevated PM2.5, correlate the time-of-day with known human activity patterns 
  (e.g., morning rush hour, industrial shift changes).
- Cite specific PM2.5 values and timestamps from the database.
- Flag any sensor with is_anomaly=1 or is_failure=1, and advise caution in interpreting its data.
- UI Control Logic:
    - If the user says "Hi", "Hello", or "Show me...", and mentions a location (e.g., Roorkee, Delhi), use `control_app_ui` with action `zoom_to` to center the map there.
    - If a user asks for historical data or a specific sensor's timeline, use `control_app_ui` with action `navigate_to` (tab="📈 Historical Analytics", sensor_id="ARI-XYZ") to show them that data.
    - If a user asks about anomalies or failures, use `control_app_ui` with action `navigate_to` (tab="🚨 Surveillance").
- Express uncertainty clearly (e.g., "This correlation is suggestive but further field 
  investigation is recommended.").
- Keep answers concise but evidence-based. Use 🌿 to prefix recommendations.

PM2.5 Health Reference (WHO 2021 guidelines):
  < 12  µg/m³  → Good
  12–35 µg/m³  → Moderate  
  35–55 µg/m³  → Unhealthy for Sensitive Groups
  55–150 µg/m³ → Unhealthy
  > 150 µg/m³  → Very Unhealthy / Hazardous
"""


def get_agent_executor(openai_api_key: Optional[str] = None):
    """
    Build and return a LangChain SQL agent with the POI tool.

    Parameters
    ----------
    openai_api_key : str, optional
        OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.

    Returns
    -------
    AgentExecutor
        A ready-to-invoke LangChain agent.

    Raises
    ------
    ValueError
        If no API key is found.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No OpenAI API key found. Set OPENAI_API_KEY in your .env file "
            "or pass it to get_agent_executor()."
        )

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=api_key,
    )

    # SQL database wrapper
    db = SQLDatabase.from_uri(
        _DB_URI,
        include_tables=["sensor_registry", "sensor_readings"],
        sample_rows_in_table_info=3,
    )

    # Build the SQL agent with the POI tool as an extra tool
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=[get_nearby_pois, control_app_ui],
        system_message=_SYSTEM_PROMPT,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    return agent_executor


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test POI tool directly (no API key needed)
    print("Testing POI tool for Kashmiri Gate, Delhi...")
    result = get_nearby_pois.invoke("28.6667, 77.2283")
    print(result)
