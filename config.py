"""全局模型配置：所有脚本统一从这里拿模型，换模型只改这一处。"""

from langchain_ollama import ChatOllama

MODEL = ChatOllama(model="qwen3.5:9b", num_ctx=32768, temperature=0)
