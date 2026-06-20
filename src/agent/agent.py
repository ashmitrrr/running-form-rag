import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make 'src' importable

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent

from storage.db import query_session, find_form_drops


# tools:

@tool
def get_form_drops(session_id: str, threshold: float = 85.0) -> str:
    """
    Find time windows in a running session where the form score dropped below a
    threshold (default 85). Use this when the user asks where/when their running
    form got worse, dropped, or degraded during a run.
    Returns the windows with their start time (seconds) and form score.
    """
    drops = find_form_drops(session_id, threshold=threshold)
    if drops.empty:
        return f"No windows below form score {threshold} in session '{session_id}'."
    return drops.to_string(index=False)


@tool
def get_session_summary(session_id: str) -> str:
    """
    Get all per-window form data for a running session: window number, start time
    in seconds, and form score for each window. Use this when the user asks for an
    overview of a run, how their form looked across the whole session, or the
    overall trend. Returns a table of every window.
    """
    df = query_session(session_id)
    if df.empty:
        return f"No data found for session '{session_id}'."
    cols = ["window", "t_start_sec", "form_score"]
    return df[cols].to_string(index=False)


@tool
def list_sessions() -> str:
    """
    List all running sessions stored in the database. Use this when the user asks
    what runs/sessions are available, or when you need to know valid session IDs.
    """
    df = query_session()
    if df.empty:
        return "No sessions stored yet."
    sessions = df["session_id"].unique()
    return "Available sessions: " + ", ".join(sessions)


# agent:

SYSTEM_PROMPT = """You are a running form analysis assistant. You help the user
understand their running biomechanics data.

You have tools to query a database of running sessions. Each session is split into
time windows, and each window has a 'form score' from 0-100 (higher = better, more
consistent form relative to the start of that run).

When the user asks about their form, use the tools to get real data, then explain
it clearly in plain language. Always ground your answers in the actual numbers the
tools return. If you don't know which session they mean, use list_sessions first.
Keep answers concise and specific."""


def build_agent():
    llm = ChatOllama(model="qwen3:14b", temperature=0)
    tools = [get_form_drops, get_session_summary, list_sessions]
    agent = create_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent


def main():
    agent = build_agent()
    print("Running form analysis agent. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue

        result = agent.invoke({"messages": [("user", user_input)]})
        # agents ans
        answer = result["messages"][-1].content
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()