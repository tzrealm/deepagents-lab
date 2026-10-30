"""阶段 0：最小可运行的 DeepAgent。

运行：python 00_first_agent.py
观察：1) 消息里出现 write_file 的 tool call；2) 磁盘上找不到文件（默认后端在内存）。
"""

from deepagents import create_deep_agent
from config import MODEL

agent = create_deep_agent(
    model=MODEL,
    system_prompt="你是一名严谨的研究助理，回答用中文。",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "用三句话解释什么是 Agent Harness，然后写进 notes/harness.md",
            }
        ]
    }
)

print("\n=== 完整消息流 ===")
for msg in result["messages"]:
    msg.pretty_print()

print("\n=== 自检 ===")
tool_names = [
    call["name"]
    for msg in result["messages"]
    for call in (getattr(msg, "tool_calls", None) or [])
]
print("调用过的工具：", tool_names or "（没有工具调用，模型可能直接回答了）")
print("磁盘上有没有 notes/ 目录？", "有（不应该）" if __import__("os").path.exists("notes") else "没有 —— 这就是 StateBackend 的行为")
