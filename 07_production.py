"""阶段 4：流式 + 持久化 + 结构化输出。

运行：python 07_production.py
观察：1) token 级流式输出；2) 同一 thread_id 第二轮能接上上文；3) 结构化输出是 Pydantic 对象。
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from deepagents import create_deep_agent


class ResearchBrief(BaseModel):
    """调研简报的结构化输出。"""

    title: str = Field(description="简报标题")
    key_points: list[str] = Field(description="3-5 条核心结论")
    open_questions: list[str] = Field(default_factory=list, description="尚未解决的问题")
    confidence: float = Field(ge=0, le=1, description="对结论的置信度")


agent = create_deep_agent(
    model="openai:gpt-5.5",
    system_prompt="你是研究助理。先列要点，再收敛结论。",
    checkpointer=InMemorySaver(),  # 生产换 PostgresSaver，否则一重启全丢
    store=InMemoryStore(),         # 生产换 PostgresStore
    response_format=ResearchBrief,
)

config = {"configurable": {"thread_id": "prod-demo-001"}}

print("=== 1. 流式（token 级）===")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "用三句话讲讲上下文工程的四类操作"}]},
    config=config,
    stream_mode="messages",
):
    # stream_mode="messages" 返回 (message_chunk, metadata) 元组
    msg_chunk, _meta = chunk
    if getattr(msg_chunk, "content", None):
        print(msg_chunk.content, end="", flush=True)
print()

print("\n=== 2. 同一 thread 的第二轮（应该记得上文）===")
second = agent.invoke(
    {"messages": [{"role": "user", "content": "刚才那四类操作，哪一类最容易被忽略？"}]},
    config=config,
)
print(second["messages"][-1].content[:400])

print("\n=== 3. 结构化输出 ===")
third = agent.invoke(
    {"messages": [{"role": "user", "content": "给『是否该用 DeepAgents』做个决策简报"}]},
    config=config,
)
brief = third.get("structured_response")
print("类型：", type(brief).__name__)
print(brief)

print("\n=== 4. 换个 thread_id 应该是全新会话 ===")
fresh = agent.invoke(
    {"messages": [{"role": "user", "content": "我们刚才聊了什么？"}]},
    config={"configurable": {"thread_id": "prod-demo-002"}},
)
print(fresh["messages"][-1].content[:300])
