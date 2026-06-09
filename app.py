
import gradio as gr
import uuid
import os
from graph_workflow import run_agent, compiled_graph

def chat(user_message, history, session_id):
    if not user_message.strip():
        return history, "", session_id
    response = run_agent(user_message, thread_id=session_id)
    history.append([user_message, response])
    return history, "", session_id

def new_session():
    return [], str(uuid.uuid4())

EXAMPLES = [
    "What are the current grow room conditions?",
    "Predict yield for the next 7 days using the LSTM model.",
    "What actuator adjustments does the PPO agent recommend?",
    "What is the outdoor weather in Karachi right now?",
    "What is the optimal CO2 level for Pleurotus ostreatus fruiting?",
    "My humidity is dropping — what should I do?",
    "Compare current temperature with the optimal fruiting range.",
    "Give me a full grow room status report.",
]

with gr.Blocks(
    title="Digital Twin Grow Room Assistant",
    theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="teal", neutral_hue="slate"),
    css="#chatbot { height: 480px; } footer { display: none !important; }",
) as demo:
    session_id = gr.State(value=str(uuid.uuid4()))
    gr.HTML("""
    <div style="text-align:center; padding:16px 0 8px">
        <h1 style="font-size:1.6rem; font-weight:700; color:#065f46">🍄 Digital Twin Grow Room Assistant</h1>
        <p style="color:#64748b">Real-time IoT monitoring · LSTM yield forecasting · PPO actuator control<br>
        <em>Pleurotus ostreatus cultivation research system</em></p>
    </div>""")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chatbot", label="Grow Room Assistant", bubble_full_width=False)
            with gr.Row():
                msg_box = gr.Textbox(placeholder="Ask about grow room conditions, yield forecast, actuator recommendations...", show_label=False, scale=5, container=False)
                send_btn = gr.Button("Send ▶", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("🗑 New Session", variant="secondary", size="sm")
                session_display = gr.Textbox(show_label=False, interactive=False, scale=3, container=False, placeholder="session id")
        with gr.Column(scale=1):
            gr.Markdown("### 🛠 Available Tools")
            gr.Markdown("|Tool|Layer|
|---|---|
|`get_environment_state`|DT Layer 1|
|`get_yield_forecast`|DT Layer 3|
|`get_rl_recommendation`|DT Layer 3|
|`get_ambient_weather`|External API|
|`search_cultivation_knowledge`|External API|")
            gr.Markdown("### 💡 Try These Prompts")
            for example in EXAMPLES:
                ex_btn = gr.Button(example, size="sm", variant="secondary")
                ex_btn.click(fn=lambda e=example: e, outputs=msg_box)
            gr.Markdown("### 📊 System Info")
            gr.Markdown("- **LLM:** Llama 3.3 70B (Groq)\n- **Framework:** LangGraph ReAct\n- **Memory:** MemorySaver\n- **Phase:** I (synthetic data)")
    send_btn.click(fn=chat, inputs=[msg_box, chatbot, session_id], outputs=[chatbot, msg_box, session_id])
    msg_box.submit(fn=chat, inputs=[msg_box, chatbot, session_id], outputs=[chatbot, msg_box, session_id])
    clear_btn.click(fn=new_session, outputs=[chatbot, session_id])
    session_id.change(fn=lambda s: f"session: {s[:8]}...", inputs=session_id, outputs=session_display)
    gr.Markdown("<center><small>Digital Twin Framework · Iqra University · MS Thesis Research</small></center>")

if __name__ == "__main__":
    demo.launch(share=True, debug=True, show_error=True)
