# Experiment Log

| Clip | Frames | FPS | Ankle conf (mean / min) | Cadence (spm) | Trunk lean | Knee @ strike | Vert osc (px) | Arm sym delta |
|------|--------|-----|-------------------------|---------------|------------|---------------|---------------|---------------|
| test_clip | 300 | 30.0 | 0.90 / 0.59 | 186.0 | 8.7° | 133.7° | 28.7 | 3.7° |
| onspot_run | 2964 | 29.88 | 0.94 / 0.82 | 166.3 (L136/R139) | 3.1° | 143.7° | 110.3 | 20.9° |

## Form score (onspot_run)
5s windows. Relative score vs first-3-window baseline. Starts 98, declines to low-80s in back half. Drops below 85: windows 10,11,15,16,18 (lowest 78.7 @ 79.8s). Driven mainly by trunk lean.

## Pipeline status
- Pose extraction → biomechanics → form score → LanceDB store → LangGraph agent (qwen3:14b) all working end-to-end.
- Agent answers natural-language queries ("where did my form drop?") correctly against stored sessions.

## Terminal Session with the agent:

((.venv) ) ashmitraina@Ashmits-MacBook-Air-2 running-form-rag % python src/agent/agent.py
/Users/ashmitraina/Documents/running-form-rag/src/agent/agent.py:78: LangGraphDeprecatedSinceV10: create_react_agent has been moved to `langchain.agents`. Please update your import to `from langchain.agents import create_agent`. Deprecated in LangGraph V1.0 to be removed in V2.0.
  agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
Running form analysis agent. Type 'quit' to exit.
You: what sessions do I have?
Agent: You have one session available: **onspot_run_2026_06_17** (looks like a run from June 17, 2026!). Would you like me to show you details about this run, like form trends or where your form dropped?
You: give me a summary of onspot_run_2026_06_17
Agent: Your run on 2026-06-17 showed a generally declining form trend over time, with notable drops:
- **Early stability**: Form started strong (98.0 at 0s) and remained above 93 for the first 19.9 seconds.
- **First major drop**: Form fell to **81.5 at 49.9s** (window 10) and stayed below 85 until window 17 (84.1 at 89.8s).
- **Second drop**: Form dipped again to **78.7 at 79.8s** (window 16), then partially recovered to 87.9 at 84.8s (window 17).
The overall pattern suggests fatigue or form degradation occurred in the second half of the run. Would you like help analyzing why this happened?
You: where did my form drop in onspot_run_2026_06_17?
Agent: Your form dropped below 85 in these time windows during **onspot_run_2026_06_17**:
- **Window 10** (49.9 seconds into the run): 81.5  
- **Window 11** (54.9 seconds): 81.9  
- **Window 15** (74.8 seconds): 81.2  
- **Window 16** (79.8 seconds): 78.7  
- **Window 18** (89.8 seconds): 84.1  
The lowest score was **78.7** in window 16 (around 80 seconds into the run). Would you like to see a comparison with other sessions or more details about these windows?
You: quit
