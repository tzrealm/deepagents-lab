"""阶段 2：双代理流水线（researcher → writer）。

运行：python 02_subagents.py
观察：LangSmith 里能看到主代理下面挂两个子 run，各自独立上下文。
踩坑：把 researcher 的 tools 删掉，它会继承主代理工具，开始自己写文件。
"""

from deepagents import create_deep_agent
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """检索网页。返回标题 + 摘要 + 链接的文本块。"""
    # 换成 Tavily / Serper / 自建检索，接口保持一致即可
    return (
        f"[MOCK 检索结果] query={query}\n"
        "1. LangChain 官方博客《Context Engineering for Agents》- 讲上下文四类操作：写、选、压、隔。\n"
        "2. Deep Agents 文档《Backends》- 六种后端，默认 StateBackend 存内存。\n"
        "3. 论文《ReAct》- 推理与行动交替的原始范式。"
    )


researcher = {
    "name": "researcher",
    "description": (
        "负责检索与事实核查。当任务需要外部事实、数据、来源对比时调用。"
        "只返回要点，不写正文。"
    ),
    "system_prompt": "你是研究员。输出格式：每条事实一行，末尾附来源。禁止推测。",
    "tools": [web_search],  # 显式收窄：不给文件工具，避免它抢 writer 的活
}

writer = {
    "name": "writer",
    "description": (
        "负责把已有素材写成结构化成稿。当素材齐备、需要产出正文时调用。不负责检索。"
    ),
    "system_prompt": "你是技术写作者。只基于给定素材写作，缺信息就标注【待补】，不得编造。",
}

agent = create_deep_agent(
    model="openai:gpt-5.5",
    system_prompt=(
        "你是主编。工作流固定：\n"
        "1) 先派 researcher 收集事实；\n"
        "2) 再派 writer 成稿；\n"
        "3) 最后自己审校，检查是否有编造内容。\n"
        "每步产物写入 /drafts/ 下。"
    ),
    subagents=[researcher, writer],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "写一篇 600 字介绍 Agent 上下文工程的短文，要提到四类上下文操作。",
            }
        ]
    }
)

print("\n=== 主代理最终输出 ===")
print(result["messages"][-1].content)

print("\n=== 派单记录 ===")
for msg in result["messages"]:
    for call in getattr(msg, "tool_calls", None) or []:
        if call["name"] == "task":
            print(f"  → {call['args'].get('subagent_type')}: {str(call['args'])[:120]}")
