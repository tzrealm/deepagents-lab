"""阶段 3：rubric 自评迭代。

运行：python 06_rubric.py
观察：agent 第一次写完会被 grader 判 needs_revision，意见塞回去后它继续改，直到 satisfied。
注意：grader 是一次额外 LLM 调用，成本翻倍，别在便宜任务上开。
"""

from deepagents import create_deep_agent
from deepagents.middleware.rubric import RubricMiddleware

agent = create_deep_agent(
    model="openai:gpt-5.5",
    system_prompt="你是技术写作者。",
    middleware=[RubricMiddleware(model="openai:gpt-5.5", max_iterations=3)],
)

RUBRIC = (
    "必须同时满足以下三条才算通过：\n"
    "1) 讲清 DeepAgents 与 LangGraph、create_agent 的分层关系；\n"
    "2) 包含至少一个可运行的代码片段；\n"
    "3) 正文 600-1000 字。"
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "写一份 800 字的 DeepAgents 技术说明"}],
        "rubric": RUBRIC,  # rubric 通过 invoke 的 state 传入，不是构造参数
    }
)

print("\n=== 最终输出 ===")
print(result["messages"][-1].content)

print("\n=== 验收状态 ===")
status = result.get("_rubric_status")
print("status =", status)
if status != "satisfied":
    print("注意：未通过时中间件不会改最后的 AIMessage，")
    print("     生产代码必须自己检查 _rubric_status / on_evaluation 回调，")
    print("     否则会把『没验收合格』的结果当成功返回给用户。")

print("\n=== 迭代轨迹 ===")
for i, msg in enumerate(result["messages"]):
    role = msg.__class__.__name__
    text = (msg.content if isinstance(msg.content, str) else str(msg.content))[:80].replace("\n", " ")
    print(f"  {i:02d} {role:18s} {text}")
