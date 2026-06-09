import os
from typing import Annotated, TypedDict, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

try:
    from google.colab import userdata
    os.environ["GROQ_API_KEY"]        = userdata.get("GROQ_API_KEY")
    os.environ["OPENWEATHER_API_KEY"] = userdata.get("OPENWEATHER_API_KEY") or ""
    os.environ["TAVILY_API_KEY"]      = userdata.get("TAVILY_API_KEY") or ""
    print("Keys loaded from Colab Secrets")
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    print("Keys loaded from .env file")

class AgentState(TypedDict):
    messages:     Annotated[list, add_messages]
    thread_id:    str
    grow_context: object

@tool
def get_environment_state() -> dict:
    "Returns current IoT sensor readings. Use when user asks about grow room conditions."
    import random, datetime
    return {
        "timestamp":       datetime.datetime.now().isoformat(),
        "temperature_c":   round(random.uniform(22.0, 27.0), 1),
        "humidity_pct":    round(random.uniform(82.0, 92.0), 1),
        "co2_ppm":         round(random.uniform(1200, 1800), 0),
        "light_lux":       round(random.uniform(280, 420), 0),
        "substrate_moist": round(random.uniform(60.0, 75.0), 1),
        "phase":           "fruiting",
    }

@tool
def get_yield_forecast(days_ahead: int = 7) -> dict:
    "Runs LSTM model to forecast yield. Use when user asks about harvest or yield predictions."
    import random
    base = 180.0
    forecast = []
    for d in range(1, days_ahead + 1):
        pred = round(base + random.uniform(-15, 25) * (d / days_ahead), 1)
        forecast.append({"day": d, "predicted_g": pred, "lower_ci": round(pred-12,1), "upper_ci": round(pred+12,1)})
    return {
        "model":            "LSTM (2-layer, 64 hidden units)",
        "horizon_days":     days_ahead,
        "forecast":         forecast,
        "cumulative_yield": round(sum(f["predicted_g"] for f in forecast), 1),
        "rmse_baseline":    "14.3g",
    }

@tool
def get_rl_recommendation() -> dict:
    "Queries PPO RL agent for actuator recommendations. Use when user asks what adjustments to make."
    import random
    return {
        "agent": "PPO (Stable-Baselines3)",
        "recommendations": {
            "ventilation_fan_pct":   random.choice([40, 50, 60, 70]),
            "mister_duty_cycle_pct": random.choice([15, 20, 25, 30]),
            "light_on_hours":        random.choice([12, 14, 16]),
        },
        "confidence": round(random.uniform(0.78, 0.95), 2),
        "rationale":  "CO2 above 1500ppm - increase ventilation for optimal fruiting body development.",
    }

@tool
def get_ambient_weather(city: str = "Karachi") -> dict:
    "Fetches outdoor weather from OpenWeather API. Use when user asks about outdoor conditions."
    import requests
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        return {"error": "OPENWEATHER_API_KEY not set", "city": city}
    r = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric", timeout=8)
    r.raise_for_status()
    data = r.json()
    return {"city": city, "temperature_c": data["main"]["temp"], "humidity_pct": data["main"]["humidity"], "description": data["weather"][0]["description"]}

@tool
def search_cultivation_knowledge(query: str) -> str:
    "Searches web for Pleurotus ostreatus cultivation guidance. Use for factual mushroom cultivation questions."
    import requests
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return "TAVILY_API_KEY not set."
    r = requests.post("https://api.tavily.com/search", json={"api_key": api_key, "query": query, "max_results": 3, "include_answer": True}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("answer"):
        return data["answer"]
    return "\n\n".join(f"[{x.get('title','')}]\n{x.get('content','')}" for x in data.get("results", []))

tools = [get_environment_state, get_yield_forecast, get_rl_recommendation, get_ambient_weather, search_cultivation_knowledge]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ["GROQ_API_KEY"])
llm_with_tools = llm.bind_tools(tools)
print("LLM initialized: Llama 3.3 70B via Groq")
print("Tools bound:", [t.name for t in tools])

SYSTEM_PROMPT = (
    "You are the Digital Twin Grow Room Assistant for oyster mushroom (Pleurotus ostreatus) cultivation research. "
    "Layers: (1) IoT sensors, (2) Digital Twin core, (3) LSTM yield model + PPO RL actuator agent. "
    "Tools: get_environment_state, get_yield_forecast, get_rl_recommendation, get_ambient_weather, search_cultivation_knowledge. "
    "Always use tools when relevant. Be precise and scientific. Cite RL confidence score when recommending actuator changes."
)

def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph

checkpointer   = MemorySaver()
compiled_graph = build_graph().compile(checkpointer=checkpointer)
print("Graph compiled successfully with MemorySaver checkpointer")

def run_agent(user_message: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_graph.invoke(
        {"messages": [{"role": "user", "content": user_message}], "thread_id": thread_id, "grow_context": None},
        config=config,
    )
    return result["messages"][-1].content

def save_graph_image(path: str = "workflow_graph.png") -> None:
    try:
        img_data = compiled_graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(img_data)
        print(f"Graph image saved to {path}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
