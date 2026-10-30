# deepagents-lab

English | [简体中文](README.zh-CN.md)

Companion exercise code for the *DeepAgents Learning Plan*. Each script maps to one stage of the main document: get it running, break it, then fix it.

## Setup

```bash
uv init --python 3.12
uv add deepagents langchain-ollama
# or
pip install -r requirements.txt

ollama pull qwen3.5:9b            # local model; install and start Ollama first
export LANGSMITH_TRACING=true      # optional but strongly recommended
export LANGSMITH_API_KEY="lsv2_..."
```

All scripts get their model from the global `MODEL` in `config.py` (default: `ChatOllama(model="qwen3.5:9b", num_ctx=32768, temperature=0)`). To switch models, change it in that one place.

## Script Overview

| Script | Stage | What you learn | Key observation |
|---|---|---|---|
| `00_first_agent.py` | 0 | Minimal runnable agent | Tool calls happen, but no files land on disk |
| `01_filesystem.py` | 1 | Real disk + virtual root boundary | Output lands in `workspace/notes/`; out-of-bounds access is blocked |
| `02_subagents.py` | 2 | Two-subagent pipeline | Two independent child runs in the trace |
| `03_skills.py` | 2 | On-demand SKILL.md loading | System prompt only contains name + description |
| `04_guardrails.py` | 3 | Permission deny + approval interrupt | Reading secrets is denied; writing to prod requires approval |
| `05_memory.py` | 3 | AGENTS.md memory | Output style changes after editing the file |
| `06_rubric.py` | 3 | Self-evaluation loop | Grader rejects once, then the agent rewrites |
| `07_production.py` | 4 | Streaming / persistence / structured output | Second turn in the same thread remembers context |
| `08_capstone.py` | 5 | Capstone project | Answers cite sources; docs are read-only |
| `09_context_overflow.py` | 1 | Actually overflow the context | Four experiments: summarization trigger / overflow fallback / unguarded crash / cumulative blowup |

`09` is special: **it costs nothing by default** — it runs the four experiments offline with a scripted fake model. Pass `--real` to call a real model.

## Suggested Practice

1. **Run it as-is first**, and check against the "Key observation" in the main document that you see the same behavior.
2. **Then change one parameter**, predict the outcome, and verify. For example:
   - `01`: set `virtual_mode` to `False` and see whether out-of-bounds writes succeed
   - `02`: remove the researcher's `tools` and see whether it starts taking over others' work
   - `04`: move the allow rule before the deny rule and see whether blocking stops working
   - `06`: set `max_iterations` to 1 and see what's returned when the rubric isn't met
3. **Finally, rewrite it without looking at the code**, and only look back when you get stuck.

## Directory Layout

```
deepagents-lab/
├── 00_first_agent.py
├── 01_filesystem.py
├── 02_subagents.py
├── 03_skills.py
├── 03_skills/skills/weekly-report/SKILL.md   # skill example
├── 04_guardrails.py
├── 05_memory.py
├── 05_memory/AGENTS.md                        # memory example
├── 06_rubric.py
├── 07_production.py
├── 08_capstone.py
├── 09_context_overflow.py
├── requirements.txt
├── workspace/          # output of 01 (generated at runtime)
├── workspace_ctx/      # output of 09 (generated at runtime; safe to delete)
└── capstone/docs/      # test docs for 08 (generated at runtime)
```
