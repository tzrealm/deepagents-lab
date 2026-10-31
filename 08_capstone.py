"""阶段 5：综合项目骨架 —— 技术文档问答 agent。

组合了：FilesystemBackend + 双子代理 + Skill + 权限 + 检查点。
运行：uv run python 08_capstone.py
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from config import MODEL
from langgraph.checkpoint.memory import InMemorySaver

DOCS_DIR = Path("./capstone/docs").resolve()
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 放几篇测试文档进去（真实项目替换成你自己的文档目录）
if not any(DOCS_DIR.glob("*.md")):
    (DOCS_DIR / "backend.md").write_text(
        "# 后端\n\n默认 StateBackend 把文件存在 agent state 里，不落盘。\n"
        "要落盘用 FilesystemBackend(root_dir=...)。\n", encoding="utf-8"
    )
    (DOCS_DIR / "subagent.md").write_text(
        "# 子代理\n\n子代理在独立上下文窗口运行，只把结论回传主代理。\n"
        "子代理默认继承父代理工具，需要显式收窄。\n", encoding="utf-8"
    )

indexer = {
    "name": "indexer",
    "description": "负责扫描文档目录、建立关键词到文件的映射。当需要先摸清文档结构时调用。",
    "system_prompt": "你是索引员。用 ls/glob/read_file 遍历文档，输出『关键词 -> 文件路径』的清单。",
    # 不传 tools：继承主代理的文件工具（只读场景可接受）
}

answerer = {
    "name": "answerer",
    "description": "负责基于索引回答具体问题。当索引已就绪、用户提出具体问题时调用。",
    "system_prompt": (
        "你是文档答疑员。只依据文档内容回答，每个结论后必须附『文件路径:行号』。"
        "文档里没有的，直接说『文档未覆盖』。"
    ),
}

agent = create_deep_agent(
    model=MODEL,
    system_prompt=(
        "你是技术文档助手。流程：\n"
        "1) 先派 indexer 建立索引，存到 /index.md；\n"
        "2) 再派 answerer 回答用户问题；\n"
        "3) 你自己检查每个结论是否都有出处。\n"
        "禁止修改 /docs/ 下任何文件。"
    ),
    subagents=[indexer, answerer],
    backend=FilesystemBackend(root_dir=DOCS_DIR.parent, virtual_mode=True),
    permissions=[
        # 文档只读：任何写操作都拒绝
        FilesystemPermission(operations=["write"], paths=["/docs/**"], mode="deny"),
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "capstone-001"}}

print("=== 第一轮：建立索引并回答 ===")
r1 = agent.invoke(
    {"messages": [{"role": "user", "content": "DeepAgents 默认后端是什么？文件存在哪？"}]},
    config=config,
)
print(r1["messages"][-1].content)

print("\n=== 第二轮：追问（验证记忆延续）===")
r2 = agent.invoke(
    {"messages": [{"role": "user", "content": "那要落盘该怎么做？给出代码。"}]},
    config=config,
)
print(r2["messages"][-1].content)

print("\n=== 验收自查 ===")
print("1. 答案里有没有『文件路径:行号』格式的出处？")
print("2. /docs/ 下的文件有没有被改动？")
print("3. 第二轮有没有用到第一轮建的 /index.md？")
