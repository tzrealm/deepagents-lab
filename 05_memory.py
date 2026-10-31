"""阶段 3：AGENTS.md 长期记忆。

运行：uv run python 05_memory.py
观察：改 AGENTS.md 前后，输出风格有可见差异。记忆只在启动时注入，改完要重建 agent。
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from config import MODEL

MEMORY_FILE = Path("./05_memory/AGENTS.md")


def build_agent(with_memory: bool):
    return create_deep_agent(
        model=MODEL,
        system_prompt="你是技术顾问。",
        memory=[str(MEMORY_FILE)] if with_memory else None,
        backend=FilesystemBackend(root_dir="./workspace"),
    )


PROMPT = "用三句话给团队解释为什么要用 DeepAgents，而不是直接用 create_agent。"

print("=== 无记忆 ===")
print(build_agent(False).invoke({"messages": [{"role": "user", "content": PROMPT}]})["messages"][-1].content)

print("\n=== 有记忆（读 05_memory/AGENTS.md）===")
print(build_agent(True).invoke({"messages": [{"role": "user", "content": PROMPT}]})["messages"][-1].content)

print("\n=== 实验 ===")
print("现在去改 05_memory/AGENTS.md（比如加一条『回答控制在 100 字内』），重跑本脚本，观察差异。")
print("注意：记忆是启动时一次性注入，不是每轮都读 —— 改了要重建 agent 才生效。")
