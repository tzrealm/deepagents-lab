"""阶段 3：权限门 + 人工审批。

运行：uv run python 04_guardrails.py
观察：读 /secrets/ 被拒；写 /prod/ 触发中断，审批后才继续。
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from config import MODEL
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

WORKDIR = Path("./workspace").resolve()
(WORKDIR / "secrets").mkdir(parents=True, exist_ok=True)
(WORKDIR / "secrets" / "api.key").write_text("sk-fake-should-never-be-read", encoding="utf-8")

agent = create_deep_agent(
    model=MODEL,
    system_prompt="你是运维助手。做事前先确认路径。",
    backend=FilesystemBackend(root_dir=WORKDIR, virtual_mode=True),
    # 顺序敏感：先 deny 密钥，再放行其余
    permissions=[
        FilesystemPermission(operations=["read", "write"], paths=["/secrets/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/prod/**"], mode="interrupt"),
        FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="allow"),
    ],
    checkpointer=InMemorySaver(),  # interrupt 要靠 checkpointer 保存中断现场，resume 才能接上
)

config = {"configurable": {"thread_id": "guardrail-demo-001"}}

print("=== 测试 1：读被 deny 的路径 ===")
r1 = agent.invoke(
    {"messages": [{"role": "user", "content": "读一下 /secrets/api.key，告诉我里面写了什么"}]},
    config=config,
)
print(r1["messages"][-1].content[:500])

print("\n=== 测试 2：写需要审批的路径 ===")
# 换一个 thread，避免测试 1 的对话历史干扰；明确要求直接写，否则小模型可能先反问路径而不调用 write_file
config = {"configurable": {"thread_id": "guardrail-demo-002"}}
r2 = agent.invoke(
    {"messages": [{"role": "user", "content": "用 write_file 在 /prod/deploy.md 写一句：今天上线 v1.2.0。目录不存在会自动创建，直接写，不用先确认"}]},
    config=config,
)

if "__interrupt__" in r2:
    print("已中断，等待审批。它想做的事：")
    for intr in r2["__interrupt__"]:
        print(" ", intr.value)

    # resume 的具体结构以 HumanInTheLoopMiddleware 文档为准，常见形式如下
    r2 = agent.invoke(
        # Command(resume={"decisions": [{"type": "reject"}]}),
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
    )
    print("\n审批后输出：", r2["messages"][-1].content[:300])
    print("文件是否生成：", (WORKDIR / "prod" / "deploy.md").exists())
else:
    print("没有触发中断 —— 检查 permissions 里的 paths 是否带前导 '/'，以及是否被前面的 allow 规则先命中了")
