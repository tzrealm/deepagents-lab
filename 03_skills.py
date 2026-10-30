"""阶段 2：加载自定义 Skill，验证渐进式披露。

运行：python 03_skills.py
观察：系统提示里只有 name + description 两行；说"写周报"时才会去读 SKILL.md 全文。
"""

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model="openai:gpt-5.5",
    system_prompt="你是助理。严格按可用技能的流程执行。",
    skills=["./03_skills/skills/"],  # 注意：是目录，不是文件
    backend=FilesystemBackend(root_dir="."),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "帮我写本周周报。素材：\n"
                    "- 完成了 DeepAgents 学习计划的阶段 1 和 2\n"
                    "- 修了 3 个线上 bug，平均响应时间从 800ms 降到 320ms\n"
                    "- 阻塞：等法务确认数据合规口径"
                ),
            }
        ]
    }
)

print("\n=== 输出 ===")
print(result["messages"][-1].content)

print("\n=== 技能加载记录（看它有没有读 SKILL.md）===")
for msg in result["messages"]:
    for call in getattr(msg, "tool_calls", None) or []:
        if call["name"] in {"read_file", "glob", "grep", "ls"}:
            print(f"  {call['name']}: {str(call['args'])[:150]}")
