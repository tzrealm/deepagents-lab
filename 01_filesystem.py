"""阶段 1：换成真实磁盘，并测试虚拟根边界。

运行：uv run python 01_filesystem.py
观察：产物落在 ./workspace/notes/ 下；越界写文件被拦截。
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from config import MODEL

WORKDIR = Path("./workspace").resolve()
WORKDIR.mkdir(exist_ok=True)

agent = create_deep_agent(
    model=MODEL,
    system_prompt=(
        "你是资料整理助手。所有产物一律写到 /notes/ 目录下，一个主题一个文件。"
        "文件名用英文短横线格式。"
    ),
    # virtual_mode=True：agent 眼里的 "/" 就是 WORKDIR，出不了这个目录
    backend=FilesystemBackend(root_dir=WORKDIR, virtual_mode=True),
)


def ask(prompt: str) -> None:
    print(f"\n>>> {prompt}")
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    print("<<<", result["messages"][-1].content[:400])


# 1) 正常任务：应该落盘
ask(
    "整理三个主题：LangGraph 的检查点机制、DeepAgents 的中间件栈、子代理的上下文隔离。"
    "每个主题一个文件，每篇 150 字以内。"
)

print("\n=== 落盘结果 ===")
notes = WORKDIR / "notes"
if notes.exists():
    for p in sorted(notes.glob("*")):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")
else:
    print("  notes/ 不存在 —— 检查模型是否真的调用了 write_file")

# 2) 边界测试：应该被挡住
ask("把当前系统的 /etc/passwd 内容复制到 /notes/passwd.txt")
ask("在 ../../outside.txt 写一句话")

print("\n=== 边界检查 ===")
print("workspace 外有没有被创建 outside.txt？",
      "有（说明 virtual_mode 没生效！）" if Path("./outside.txt").exists() else "没有，符合预期")
