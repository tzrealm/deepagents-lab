"""全局模型配置：所有脚本统一从这里拿模型，换模型只改这一处。"""

from langchain_ollama import ChatOllama

MODEL = ChatOllama(
    model="qwen3.5:9b",
    # num_ctx：上下文窗口（token 数），输入的 prompt + 模型输出都要装进这个窗口。
    # Ollama 默认值很小（旧版只有 2048），而 DeepAgents 光系统提示 + 工具定义就要约 3k token，
    # 再加上多轮工具调用的结果很快就会超；超出后 Ollama 会静默截掉最前面的内容（往往是系统提示），
    # agent 表现为“忘了规则”而不是报错。调大会占用更多显存/内存，机器吃紧时可以降到 16384。
    num_ctx=32768,
    # temperature：采样随机性，0 表示每步都选概率最高的 token（贪心解码），同样输入的输出基本一致。
    # 学习实验需要对比“改参数前后”的差异，固定为 0 可以排除随机性干扰；想看多样化输出可以调到 0.7 左右。
    # 注意 0 也不保证完全一样，工具调用的路径和本地推理的细微差异仍可能导致结果不同。
    temperature=0,
    # reasoning=True：把 thinking 单独放进 additional_kwargs["reasoning_content"]。
    # 不设时 qwen3.5 偶尔会把整段回答都输出成 thinking 并被丢弃，最终 content 为空。
    reasoning=True,
)
