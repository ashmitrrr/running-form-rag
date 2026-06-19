"""
Gradio UI for the local-first running form analysis agent.

Run from the PROJECT ROOT with:
    python src/app/app.py
    
"""

import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")  #for server-side plotting
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from storage.db import query_session   # noqa: E402
from agent.agent import build_agent      # noqa: E402


AGENT = build_agent()


# data helpers
def get_sessions():
    df = query_session()
    if df.empty:
        return []
    return sorted(df["session_id"].unique())


def session_stats_md(session_id):
    """Markdown block of quick stats for the selected session."""
    if not session_id:
        return "No session selected."
    df = query_session(session_id)
    if df.empty:
        return f"No data for **{session_id}**."

    avg_form = round(df["form_score"].mean(), 1)
    worst = df.loc[df["form_score"].idxmin()]
    duration = df["t_start_sec"].max() + 5

    return (
        f"### {session_id}\n"
        f"- **Avg form score:** {avg_form}\n"
        f"- **Lowest:** {worst['form_score']:.1f} at {worst['t_start_sec']:.0f}s\n"
        f"- **Windows:** {len(df)}\n"
        f"- **Duration:** ~{duration:.0f}s"
    )


def make_form_chart(session_id):
    """Matplotlib line chart of form score over the run."""
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)

    if session_id:
        df = query_session(session_id).sort_values("t_start_sec")
        if not df.empty:
            ax.plot(df["t_start_sec"], df["form_score"],
                    color="#2f9e6f", linewidth=2.4, marker="o", markersize=5)
            ax.axhline(85, color="#d1495b", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.set_ylim(60, 100)

    ax.set_xlabel("Time into run (seconds)", fontsize=9)
    ax.set_ylabel("Form score", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


# chat
def respond(message, history, session_id):
    """Send a message to the agent, with the selected session as context."""
    context = message
    if session_id:
        context = (
            f"(The user is currently looking at session '{session_id}'. "
            f"If they don't specify a session, assume this one.)\n\n{message}"
        )
    result = AGENT.invoke({"messages": [("user", context)]})
    return result["messages"][-1].content


# ui layout
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.gray,
)

with gr.Blocks(title="Running Form Analyst") as demo:
    gr.Markdown("# Running Form Analyst")
    gr.Markdown("Answers are ran through locally run model.")

    sessions = get_sessions()
    default_session = sessions[0] if sessions else None

    with gr.Row():
        # left column: session + stats + chart
        with gr.Column(scale=1):
            session_dd = gr.Dropdown(
                choices=sessions, value=default_session,
                label="Session", interactive=True,
            )
            stats_md = gr.Markdown(session_stats_md(default_session))
            chart = gr.Plot(make_form_chart(default_session), label="Form score over the run")

        # right column: chat
        with gr.Column(scale=2):
            gr.ChatInterface(
                fn=respond,
                additional_inputs=[session_dd],
                chatbot=gr.Chatbot(height=460),
            )

    # when session changes, refresh stats + chart
    session_dd.change(
        fn=lambda s: (session_stats_md(s), make_form_chart(s)),
        inputs=session_dd,
        outputs=[stats_md, chart],
    )


if __name__ == "__main__":
    demo.launch(theme=THEME)