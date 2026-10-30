# deepagents-lab

[English](README.md) | 简体中文

《DeepAgents 学习计划》的配套练习代码。每个脚本对应主文档的一个阶段，先跑通，再改坏，再修好。

## 准备

```bash
uv init --python 3.12
uv add deepagents langchain-ollama
# 或者
pip install -r requirements.txt

ollama pull qwen3.5:9b            # 本地模型，需先安装并启动 Ollama
export LANGSMITH_TRACING=true      # 可选但强烈建议
export LANGSMITH_API_KEY="lsv2_..."
```

所有脚本统一从 `config.py` 的全局变量 `MODEL` 取模型（默认 `ChatOllama(model="qwen3.5:9b", num_ctx=32768, temperature=0)`），换模型只改这一处。

## 脚本对照表

| 脚本 | 阶段 | 学什么 | 关键观察 |
|---|---|---|---|
| `00_first_agent.py` | 0 | 最小可运行 agent | 有 tool call，但磁盘上没文件 |
| `01_filesystem.py` | 1 | 真实磁盘 + 虚拟根边界 | 产物落 `workspace/notes/`，越界被挡 |
| `02_subagents.py` | 2 | 双子代理流水线 | trace 里两个独立子 run |
| `03_skills.py` | 2 | SKILL.md 按需加载 | 系统提示只有 name+description |
| `04_guardrails.py` | 3 | 权限 deny + 审批 interrupt | 读密钥被拒，写 prod 要批 |
| `05_memory.py` | 3 | AGENTS.md 记忆 | 改文件后输出风格变化 |
| `06_rubric.py` | 3 | 自评迭代 | grader 打回一次后重写 |
| `07_production.py` | 4 | 流式/持久化/结构化输出 | 同 thread 第二轮记得上文 |
| `08_capstone.py` | 5 | 综合项目 | 答案带出处、文档只读 |
| `09_context_overflow.py` | 1 | 让上下文真爆一次 | 四个实验：摘要触发 / 溢出兜底 / 裸奔崩溃 / 累积爆掉 |

`09` 比较特殊：**默认不花钱**，用脚本化的假模型离线跑四个实验，`--real` 才调真实模型。

## 建议的练习方式

1. **先原样跑通**，对照主文档的"关键观察"确认现象一致。
2. **再改一个参数**，预测结果，然后验证。比如：
   - `01`：把 `virtual_mode` 改成 `False`，看越界写会不会成功
   - `02`：删掉 researcher 的 `tools`，看它是不是开始抢活
   - `04`：把 allow 规则挪到 deny 前面，看拦截是否失效
   - `06`：把 `max_iterations` 改成 1，看未通过时返回什么
3. **最后不看代码重写一遍**，卡住再回头看。

## 目录说明

```
deepagents-lab/
├── 00_first_agent.py
├── 01_filesystem.py
├── 02_subagents.py
├── 03_skills.py
├── 03_skills/skills/weekly-report/SKILL.md   # 技能示例
├── 04_guardrails.py
├── 05_memory.py
├── 05_memory/AGENTS.md                        # 记忆示例
├── 06_rubric.py
├── 07_production.py
├── 08_capstone.py
├── 09_context_overflow.py
├── requirements.txt
├── workspace/          # 01 的产物落这里（运行后生成）
├── workspace_ctx/      # 09 的产物落这里（运行后生成，可直接删）
└── capstone/docs/      # 08 的测试文档（运行后生成）
```
