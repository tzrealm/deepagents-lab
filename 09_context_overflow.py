"""阶段 1 · Step 1.3：让上下文爆一次（三个层次，从免费到真爆）

背景（0.7.32 源码核对）：
DeepAgents 默认有三道防线，所以"随便塞个大文件"根本爆不了：

  1. FilesystemMiddleware(tool_token_limit_before_evict=20000)
     单条工具结果 ≥ 20k tokens（约 80000 字符）→ 落盘到 /large_tool_results/{id}，
     上下文里只留一个 stub。
  2. FilesystemMiddleware(human_message_token_limit_before_evict=50000)
     单条 HumanMessage ≥ 50k tokens → 同样落盘。
  3. SummarizationMiddleware
     默认 trigger=("fraction", 0.85) / keep=("fraction", 0.10)（模型有 profile 时）
     或 trigger=("tokens", 170000) / keep=("messages", 6)（无 profile 时）。
     到阈值先摘要，不等你爆。

所以"让上下文爆"的正确姿势是：**先把防线拆掉，或者把阈值压到地板**。

本脚本三个实验：
  A. 压阈值 —— 看摘要触发（最省事，5 秒出结果）
  B. 真爆 + 兜底 —— 制造 ContextOverflowError，看中间件摘要后重试
  C. 真爆 + 无兜底 —— 关掉摘要，看原始报错长什么样

默认用假模型离线跑（不花钱、不需要 API key）。
加 --real 参数则用真实模型跑实验 A（需要 OPENAI_API_KEY 或 ANTHROPIC_API_KEY）。

用法：
    python 09_context_overflow.py            # 离线，三个实验都跑
    python 09_context_overflow.py --real     # 真实模型跑实验 A
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

from langchain.agents.middleware import AgentMiddleware
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware

WORKDIR = Path("./workspace_ctx").resolve()


class _NoSummarization(AgentMiddleware):
    """同名空中间件：把默认的 SummarizationMiddleware 整个顶掉。

    `create_deep_agent` 按 `middleware` 的 `name` 替换默认栈里的同名层，
    所以只要名字对得上，就能用空实现替换掉它。
    注意：这只是教学实验，生产里千万别这么干 —— 等于拆掉安全带。
    """

    @property
    def name(self) -> str:
        return "SummarizationMiddleware"


# --------------------------------------------------------------------------
# 假模型：按脚本回放，且能按"输入 token 超限"抛 ContextOverflowError
# --------------------------------------------------------------------------
class ScriptedModel(BaseChatModel):
    """按顺序回放 AIMessage；同时模拟 provider 的上下文超限拒绝。"""

    script: list[AIMessage] = []
    overflow_at_tokens: int | None = None  # 超过就抛 ContextOverflowError
    calls: int = 0
    summarize_calls: int = 0
    peak_tokens: int = 0
    last_len: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        self.peak_tokens = max(self.peak_tokens, count_tokens_approximately(messages))
        self.last_len = len(messages)

        # 摘要请求：中间件会把 summary_prompt 作为 HumanMessage 发过来
        last = messages[-1].content if messages else ""
        if isinstance(last, str) and ("摘要" in last or "summar" in last.lower() or "compact" in last.lower()):
            self.summarize_calls += 1
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="【摘要】前面做了若干次文件操作，结论已写入 /notes/。"))]
            )

        # 模拟 provider 的硬上限
        if self.overflow_at_tokens is not None:
            used = count_tokens_approximately(messages)
            if used > self.overflow_at_tokens:
                raise ContextOverflowError(
                    f"prompt is too long: {used} tokens > {self.overflow_at_tokens} maximum"
                )

        msg = self.script.pop(0) if self.script else AIMessage(content="完成。")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _write_tool_call(path: str, content: str, cid: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {"file_path": path, "content": content}, "id": cid}],
    )


def _big_task_script(n: int = 6, repeat: int = 20) -> list[AIMessage]:
    """模拟一个"逐条处理并落盘"的长任务，产生 2n 条消息。

    repeat 控制每条内容的大小 —— 想让上下文涨得快就调大它。
    """
    out: list[AIMessage] = []
    for i in range(n):
        out.append(_write_tool_call(f"/notes/note_{i}.md", f"第 {i} 条记录的处理结果。" * repeat, f"c{i}"))
    out.append(AIMessage(content=f"全部 {n} 条处理完毕。"))
    return out


# --------------------------------------------------------------------------
# 实验 A：把摘要阈值压到地板，看它什么时候动手
# --------------------------------------------------------------------------
def exp_a_low_trigger(real: bool = False) -> None:
    print("\n" + "=" * 72)
    print("实验 A：压低摘要阈值 —— 亲眼看到摘要发生")
    print("=" * 72)

    root = WORKDIR / "a"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    backend = FilesystemBackend(root_dir=root, virtual_mode=True)

    if real:
        model_name = "openai:gpt-5.5"
        model: Any = model_name
        summ_model: Any = model_name
    else:
        model = ScriptedModel(script=_big_task_script(6))
        summ_model = model

    summ = SummarizationMiddleware(
        model=summ_model,
        backend=backend,
        trigger=("messages", 8),   # 默认是 0.85 的上下文占比；这里压到 8 条
        keep=("messages", 4),
    )

    agent = create_deep_agent(
        model=model,
        system_prompt="你是资料整理助手。每个主题写一个文件到 /notes/。",
        backend=backend,
        middleware=[summ],          # 同名替换默认的 SummarizationMiddleware
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "处理 6 条记录，逐条写入 /notes/。"}]})

    msgs = result["messages"]
    print(f"  state 里保留的完整消息日志: {len(msgs)} 条（私有字段，摘要不改写它）")
    print(f"  模型被调用 {model.calls} 次，其中摘要调用 {getattr(model, 'summarize_calls', 0)} 次")
    print(f"  单次请求峰值 tokens: {getattr(model, 'peak_tokens', '?')}")
    if not real:
        print(f"  实际发出去的最后一次上下文条数: {getattr(model, 'last_len', '?')}")

    hist = root / "conversation_history"
    files = sorted(hist.glob("*.md")) if hist.exists() else []
    print(f"  被驱逐历史的落盘位置: {hist}")
    for f in files:
        print(f"    {f.name}  {f.stat().st_size} 字节")
        head = f.read_text(encoding="utf-8")[:200].replace("\n", " ")
        print(f"    开头: {head}")

    if files:
        print("  ✓ 摘要确实发生了：/conversation_history/*.md 是摘要时写下的原始历史备份。")
    else:
        print("  ✗ 没触发摘要：说明消息数还没到 8 条，把 trigger 调更低。")
    print("  注意：`_summarization_event` 是 PrivateStateAttr，不会出现在 invoke() 返回值里。")
    print("       要判断摘要有没有发生，看 /conversation_history/ 有没有新文件最可靠。")


# --------------------------------------------------------------------------
# 实验 B：真爆一次，看 ContextOverflowError 的兜底路径
# --------------------------------------------------------------------------
def exp_b_overflow_fallback() -> None:
    print("\n" + "=" * 72)
    print("实验 B：真爆 + 兜底 —— 制造 ContextOverflowError，看中间件救不救")
    print("=" * 72)

    def run_once(tag: str, keep_messages: int) -> None:
        root = WORKDIR / f"b_{tag}"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        backend = FilesystemBackend(root_dir=root, virtual_mode=True)

        # 关键：把摘要阈值调到一个不可能自然到达的高度，这样才会真的撞到 provider 上限
        model = ScriptedModel(script=_big_task_script(6, repeat=400), overflow_at_tokens=2000)

        summ = SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=("tokens", 10**9),      # 相当于关掉"提前摘要"，只剩溢出兜底
            keep=("messages", keep_messages),
        )
        agent = create_deep_agent(
            model=model,
            system_prompt="你是资料整理助手。",
            backend=backend,
            middleware=[summ],
        )

        print(f"\n  [{tag}] keep=('messages', {keep_messages})")
        try:
            agent.invoke({"messages": [{"role": "user", "content": "处理 6 条记录。"}]})
        except ContextOverflowError as e:
            print(f"    ✗ 兜底也失败了：{e}")
            print(f"      模型调用 {model.calls} 次，摘要调用 {model.summarize_calls} 次")
            print("      → 摘要只压'旧'消息，keep 的尾巴里还留着几条巨大的 write_file 参数，")
            print("        重试照样超限。真实场景要么把 keep 调小，要么配 truncate_args_settings。")
            return
        except Exception as e:  # noqa: BLE001
            print(f"    ✗ 其他异常: {type(e).__name__}: {e}")
            return

        hist = root / "conversation_history"
        files = sorted(hist.glob("*.md")) if hist.exists() else []
        print(f"    ✓ 没崩。模型调用 {model.calls} 次，摘要调用 {model.summarize_calls} 次")
        print(f"      历史落盘文件 {len(files)} 个，峰值 tokens {model.peak_tokens}")

    run_once("keep 太大", keep_messages=4)
    run_once("keep 收紧", keep_messages=2)
    print("\n  → 这就是 ContextOverflowError 兜底的价值：把'必然失败'变成'多花一次调用'；")
    print("    但兜底不是万能的，keep 留太多照样救不回来。")


# --------------------------------------------------------------------------
# 实验 C：拆掉防线，看裸奔报错
# --------------------------------------------------------------------------
def exp_c_no_defense() -> None:
    print("\n" + "=" * 72)
    print("实验 C：拆掉防线 —— 关掉摘要 + 关掉单条结果落盘")
    print("=" * 72)

    root = WORKDIR / "c"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    # 注意：单行超长会被 MAX_LINE_LENGTH（5000 字符）裁掉，所以必须写成多行
    big = "\n".join(f"2026-10-29 12:00:{i:02d} INFO 这是一条测试日志记录，用来把上下文撑大。" for i in range(3000))
    (root / "big.log").write_text(big, encoding="utf-8")
    print(f"  造了一个 {len(big)} 字符的日志文件（约 {len(big) // 4} tokens）")

    backend = FilesystemBackend(root_dir=root, virtual_mode=True)
    model = ScriptedModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "/big.log", "limit": 3000},  # 必须显式放大 limit
                        "id": "r1",
                    }
                ],
            ),
            AIMessage(content="读完了。"),
        ],
        overflow_at_tokens=5000,   # 故意设得比单次读取还小
    )

    # 关掉单条工具结果落盘（None = 不驱逐），并用同名空中间件把摘要层整个顶掉
    fs = FilesystemMiddleware(
        backend=backend,
        tool_token_limit_before_evict=None,      # 关键：大结果留在上下文里
        human_message_token_limit_before_evict=None,
    )
    agent = create_deep_agent(
        model=model,
        system_prompt="你是资料整理助手。",
        backend=backend,
        middleware=[fs, _NoSummarization()],
    )

    try:
        r = agent.invoke({"messages": [{"role": "user", "content": "读一下 /big.log 并总结。"}]})
        tm = [m for m in r["messages"] if type(m).__name__ == "ToolMessage"]
        ai = [m for m in r["messages"] if type(m).__name__ == "AIMessage" and getattr(m, "tool_calls", None)]
        if ai:
            print(f"  实际下发的工具参数: {ai[0].tool_calls[0]['args']}")
        print(f"  工具返回长度 {len(tm[0].content) if tm else 0} 字符")
        print(f"  返回头部: {tm[0].content[:220].replace(chr(10), ' / ') if tm else '-'}")
        print(f"  返回尾部: {tm[0].content[-260:].replace(chr(10), ' / ') if tm else '-'}")
        print("  → 没爆：说明单次读取还没到模拟上限，把 overflow_at_tokens 调小再试")
    except ContextOverflowError as e:
        print(f"  ✓ 爆了，且没人兜底：{e}")
        print("  → 生产环境里这就是用户看到的 500。实验 B 的配置就是为了避免它。")
    except Exception as e:  # noqa: BLE001
        print(f"  其他异常: {type(e).__name__}: {e}")


def exp_d_accumulate() -> None:
    """实验 D：不靠大文件，靠"多条中等结果"堆爆 —— 这才是真实爆法。"""
    print("\n" + "=" * 72)
    print("实验 D：真实爆法 —— 多条中等结果累积")
    print("=" * 72)

    root = WORKDIR / "d"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    # 每份 200 行、约 9000 字符 ≈ 2300 tokens，远低于单条 20k 的落盘阈值
    for i in range(12):
        body = "\n".join(f"第 {j} 行：这是第 {i} 份资料的内容，用来慢慢把上下文堆高。" for j in range(200))
        (root / f"doc_{i}.log").write_text(body, encoding="utf-8")

    backend = FilesystemBackend(root_dir=root, virtual_mode=True)
    script = [
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": f"/doc_{i}.log", "limit": 200}, "id": f"r{i}"}])
        for i in range(12)
    ]
    script.append(AIMessage(content="全部读完了。"))

    model = ScriptedModel(script=script, overflow_at_tokens=15000)   # 模拟 15k 的小窗口模型
    fs = FilesystemMiddleware(
        backend=backend,
        tool_token_limit_before_evict=None,          # 关掉落盘，否则每份都被换成 stub
        human_message_token_limit_before_evict=None,
    )
    agent = create_deep_agent(
        model=model,
        system_prompt="你是资料整理助手。",
        backend=backend,
        middleware=[fs, _NoSummarization()],
    )

    try:
        agent.invoke({"messages": [{"role": "user", "content": "依次读完 12 份资料。"}]})
        print(f"  没爆。峰值 tokens {model.peak_tokens}，模型调用 {model.calls} 次")
    except ContextOverflowError as e:
        print(f"  ✓ 爆了：{e}")
        print(f"    爆在第 {model.calls} 次模型调用，当时上下文约 {model.peak_tokens} tokens")
        print("  → 注意：单份资料才 2300 tokens 左右，远低于 20k 的落盘阈值，")
        print(f"    所以它一份份全留在了上下文里，读到第 {model.calls - 2} 份就撑不住了。")
        print("    这就是真实项目里'跑着跑着突然崩'的典型成因。")


if __name__ == "__main__":
    real = "--real" in sys.argv
    print(f"工作目录: {WORKDIR}")
    if real:
        print("模式: 真实模型（实验 A）")
        exp_a_low_trigger(real=True)
    else:
        print("模式: 假模型离线（不花钱）")
        exp_a_low_trigger()
        exp_b_overflow_fallback()
        exp_c_no_defense()
        exp_d_accumulate()
    print("\n" + "=" * 72)
    print("结论：上下文不是被'一个大文件'撑爆的，是被'很多条中等结果'堆爆的。")
    print("      真实顺序是五道闸依次生效：")
    print("        1. read_file 默认只读 100 行（DEFAULT_READ_LIMIT）")
    print("        2. 单行超 5000 字符被裁（MAX_LINE_LENGTH）")
    print("        3. 单条工具结果超 20k tokens → 落盘 /large_tool_results/")
    print("        4. 单条人类消息超 50k tokens → 同样落盘")
    print("        5. 上下文到 85% → 摘要；真溢出 → 摘要后重试")
    print("      想让它爆，就得主动拆掉第 3、5 道闸（实验 C/D），")
    print("      或者把第 5 道的阈值压到地板先看清它的行为（实验 A）。")
    print("=" * 72)
