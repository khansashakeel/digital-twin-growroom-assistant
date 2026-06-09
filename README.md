# 🍄 Digital Twin Grow Room Assistant

> An agentic AI assistant built with LangChain + LangGraph for real-time monitoring, yield forecasting, and actuator control of an oyster mushroom (*Pleurotus ostreatus*) cultivation system.

---

## 📌 Project Overview

This project is developed as part of an MS thesis research assignment on:

**"A Digital Twin Framework for Real-Time Environmental Monitoring and Predictive Yield Optimization of Oyster Mushroom Cultivation"**

The assistant provides a conversational interface to interact with a three-layer Digital Twin system:

| Layer | Description |
|-------|-------------|
| Layer 1 | IoT sensor network — temperature, humidity, CO2, light, substrate moisture |
| Layer 2 | Digital Twin core — real-time virtual replica of the grow room |
| Layer 3 | LSTM yield prediction model + PPO reinforcement learning actuator control agent |

---

## 🎯 Selected Use Case

**Smart Grow Room Management Assistant**

Researchers and growers can query the system in natural language to:
- Monitor real-time grow room environmental conditions
- Get LSTM-based multi-step yield forecasts
- Receive PPO RL agent actuator recommendations
- Check outdoor ambient weather for HVAC context
- Search peer-reviewed cultivation knowledge

---

## 🛠 Tools Used

| # | Tool | Type | Maps To |
|---|------|------|---------|
| 1 | `get_environment_state` | Custom Python | DT Layer 1 — IoT sensor readings |
| 2 | `get_yield_forecast` | Custom Python | DT Layer 3 — LSTM model inference |
| 3 | `get_rl_recommendation` | Custom Python | DT Layer 3 — PPO agent actions |
| 4 | `get_ambient_weather` | External API | OpenWeather API |
| 5 | `search_cultivation_knowledge` | External API | Tavily Search API |

---

## 🔌 APIs Integrated

### 1. OpenWeather API
- **Purpose:** Fetch real-time outdoor temperature and humidity
- **Use case:** Provides ambient context for HVAC and ventilation decisions
- **Free tier:** 1,000 calls/day — no credit card required
- **Endpoint:** `https://api.openweathermap.org/data/2.5/weather`

### 2. Tavily Search API
- **Purpose:** Web search for peer-reviewed cultivation guidance
- **Use case:** Answers factual questions about *Pleurotus ostreatus* cultivation
- **Free tier:** 1,000 searches/month — no credit card required
- **Endpoint:** `https://api.tavily.com/search`

### 3. Groq API (LLM)
- **Purpose:** Hosts Llama 3.3 70B — the reasoning engine of the agent
- **Free tier:** 14,400 requests/day
- **Model used:** `llama-3.3-70b-versatile`

---

## 🔄 LangGraph Workflow

The application uses a **ReAct (Reason + Act)** pattern implemented as a LangGraph `StateGraph`:

```
[START]
   │
agent_node  ◄─────────────────────────┐
   │                                   │
tools_condition                        │
 ├─ tool call detected ──► ToolNode ───┘
 └─ no tool call ──────────────────► END
```

### Nodes
| Node | Role |
|------|------|
| `agent_node` | LLM reasons about user query, decides which tools to call |
| `ToolNode` | Executes selected tools, returns results to agent |

### Edges
| Edge | Type | Condition |
|------|------|-----------|
| `agent → tools` | Conditional | LLM generated tool calls |
| `agent → END` | Conditional | No tool calls needed |
| `tools → agent` | Fixed | Always returns to agent after execution |

### State
```python
class AgentState(TypedDict):
    messages:     Annotated[list, add_messages]  # conversation history
    thread_id:    str                            # session identifier
    grow_context: object                         # sensor snapshot
```

---

## 🧠 Memory Implementation

Memory is implemented using LangGraph's `MemorySaver` checkpointer:

```python
checkpointer   = MemorySaver()
compiled_graph = build_graph().compile(checkpointer=checkpointer)
```

### How it works
- Every conversation turn is persisted as a **checkpoint** keyed by `thread_id`
- Each Gradio session generates a unique `thread_id` (UUID)
- On every new turn, the full conversation history is loaded from the checkpoint
- The agent has full context of all previous messages in the session

### Graph State vs Memory
| Concept | Description |
|---------|-------------|
| **Graph State** | Temporary data passed between nodes within one graph run (messages, grow_context) |
| **Memory / Checkpoint** | Persistent storage across multiple graph runs, keyed by thread_id |

### Multi-turn Example
```
Turn 1: "What are the grow room conditions?"     → agent reads sensors
Turn 2: "Is that humidity level safe?"           → agent remembers Turn 1 context
Turn 3: "What does the RL agent recommend?"      → agent uses full history
```

---

## 🖥 Gradio Interface

The UI is built with `gr.Blocks` using an emerald theme and includes:

- **Chat window** — multi-turn conversation with avatar
- **Send button + Enter key** support
- **New Session button** — clears history and creates fresh memory thread
- **Session ID display** — shows active thread_id
- **Tools panel** — lists all 5 tools with their DT layer
- **Example prompts** — 8 clickable prompt buttons
- **System info panel** — LLM, framework, memory, phase info

---

## 🚀 How to Run

### Prerequisites
```bash
pip install langgraph langchain langchain-groq langchain-community gradio python-dotenv requests
```

### API Keys Required
| Key | Get From |
|-----|----------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `OPENWEATHER_API_KEY` | [openweathermap.org/api](https://openweathermap.org/api) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) |

### Local Setup
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/digital-twin-growroom-assistant.git
cd digital-twin-growroom-assistant

# Copy and fill environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the app
python app.py
```

### Google Colab
```python
# Add keys to Colab Secrets (key icon in sidebar), then:
!pip install langgraph langchain langchain-groq gradio python-dotenv -q
!python app.py
```

---

## 💬 Example Prompts

| Prompt | Tools Called |
|--------|-------------|
| `What are the current grow room conditions?` | `get_environment_state` |
| `Predict yield for the next 7 days` | `get_yield_forecast` |
| `What actuator adjustments does the PPO agent recommend?` | `get_rl_recommendation` |
| `What is the outdoor weather in Karachi?` | `get_ambient_weather` |
| `What is the optimal CO2 range for Pleurotus ostreatus?` | `search_cultivation_knowledge` |
| `Give me a full grow room status report` | All 3 custom tools |
| `My humidity is dropping — what should I do?` | `get_environment_state` + `get_rl_recommendation` |
| `Compare current temperature with optimal fruiting range` | `get_environment_state` + `search_cultivation_knowledge` |

---

## 📊 Workflow Diagram

![Workflow Diagram](workflow_diagram.png)

---

## 📸 Screenshots

### Gradio UI
![Gradio UI](screenshots/gradio_ui.png)

### Tool Execution — PPO Agent Recommendation
![PPO Recommendation](screenshots/ppo_recommendation.png)

### Multi-turn Conversation
![Multi-turn](screenshots/multi_turn.png)

---

## ⚠️ Challenges Faced

1. **LLM provider compatibility** — Gemini 1.5 Flash was deprecated; switched to Groq (Llama 3.3 70B) which proved faster and more reliable
2. **API key management in Colab** — Resolved using Colab Secrets + try/except fallback to `.env`
3. **File writing in Colab** — String escaping issues when writing Python files from within Python; resolved using line-by-line list approach
4. **Tool docstrings** — LangGraph requires clear, specific docstrings for the LLM to correctly decide when to invoke each tool

---

## 🔮 Future Improvements

1. **Phase II integration** — Replace mock sensor data with real InfluxDB time-series queries
2. **Real LSTM inference** — Load trained model weights and run actual predictions
3. **Real PPO agent** — Deploy Stable-Baselines3 PPO model for live actuator recommendations
4. **LangSmith tracing** — Add observability for debugging agent reasoning chains
5. **Streaming responses** — Stream LLM output token-by-token in Gradio
6. **Multi-agent system** — Separate monitoring agent, forecasting agent, and control agent with a supervisor
7. **RAG integration** — Index thesis literature for grounded cultivation knowledge
8. **Hugging Face Spaces deployment** — Deploy for public access and demonstration
9. **Human-in-the-loop** — Add approval step before actuator changes are applied
10. **Database integration** — Log all conversations and tool outputs to SQLite for analysis

---

## 📋 Evaluation Rubric Coverage

| Component | Marks | Implementation |
|-----------|-------|---------------|
| Use case design | 10 | Thesis-aligned Digital Twin assistant |
| LangGraph workflow | 20 | ReAct StateGraph with conditional routing |
| Tool integration | 15 | 5 tools — 3 custom Python + 2 external API |
| API integration | 15 | OpenWeather + Tavily Search |
| Conditional routing | 10 | `tools_condition` built-in router |
| Memory implementation | 10 | `MemorySaver` with thread-based sessions |
| Gradio interface | 10 | Full chat UI with tools panel + example prompts |
| Documentation | 10 | This README |
| **Total** | **100** | ✅ |

---

## 👩‍💻 Author

**Khansa**
MS Software Engineering — Iqra University
Thesis: *A Digital Twin Framework for Real-Time Environmental Monitoring and Predictive Yield Optimization of Oyster Mushroom Cultivation*

---

## 📄 License

MIT License — feel free to use and adapt for research purposes.
