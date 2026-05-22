# revive-companion 💘

**让 AI 主动找你时，像真人一样"想你了"，而不是"该发消息了"。**

用泊松过程 + 贝叶斯推断 + 信息增益，决定**该不该**、**什么时候**打扰用户。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/pearthink123/revive-companion/actions/workflows/ci.yml/badge.svg)](https://github.com/pearthink123/revive-companion/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/revive-companion.svg)](https://pypi.org/project/revive-companion/)
[![Tests](https://img.shields.io/badge/tests-124%20passing-brightgreen.svg)]()

English | [中文](README_zh.md)

---

## Why Not Cron?

| 方案 | 时机决策 | 上下文感知 | 长期记忆 | 情感建模 |
|------|---------|-----------|---------|---------|
| Cron / 定时任务 | ❌ 固定间隔 | ❌ | ❌ | ❌ |
| Random | ⚠️ 纯随机 | ❌ | ❌ | ❌ |
| **revive-companion** | ✅ 泊松+贝叶斯 | ✅ 推断用户状态 | ⚠️ 概率曲线 | ❌ |
| **affective-longing** | ✅ 继承 | ✅ 继承 | ✅ ChromaDB | ✅ VAD 情绪 |

> 不是"用户活跃就多发"，而是"推断用户在做什么 → 决定该不该打扰"。

---

## Quick Start

### 极简版（仅引擎决策）

```bash
pip install revive-companion
```

```python
from revive_companion import PoissonLove

love = PoissonLove()

result = love.tick()
if result.should_send:
    # result.prompt 包含上下文信息（时间、渴望度、用户状态）
    my_send_function(result.prompt)
    love.record_send()

# 用户回复后
love.record_reply(reply_speed=0.8, reply_length=0.6)
```

### 完整版（引擎 + LLM + Runner）

```python
from revive_companion import PoissonLove
from revive_companion.adapters import OpenAIAdapter
from revive_companion.runner import Runner
from revive_companion.core.config import Config

config = Config.from_dict({
    "engagement": {"lambda_rate": 0.15, "check_interval_minutes": 30},
    "persona": {"name": "小克", "tone": "warm", "context": "你是一个关心人的AI伴侣"},
})

love = PoissonLove(config=config)
adapter = OpenAIAdapter(config, api_key="sk-...")
runner = Runner(love, adapter)

# 每30分钟自动检查，如果决定发送会调用 LLM 生成消息
runner.run()
```

---

## 三阶段决策

| 阶段 | 模块 | 核心问题 | 原理 |
|------|------|---------|------|
| 🎲 **时机** | Poisson 过程 | 什么时候考虑？ | 像真人"突然想你了"一样随机触发 |
| 📊 **价值** | 信息论 | 值不值得发？ | 如果已知用户状态，就别打扰了 |
| 🧠 **状态** | 贝叶斯推断 | 用户在干嘛？ | 推断隐藏状态 → 做对应决策 |

引擎推断的用户状态：

| 推断状态 | 发送效用 | 决策 |
|---------|---------|------|
| 🗣️ 聊天中 | 0.2 | ❌ 别打断 |
| 💻 闲逛 | 0.7 | ✅ 好时机 |
| 💼 忙碌 | 0.1 | ❌ 别打扰 |
| 😴 睡觉 | 0.0 | ❌ 永远不发 |
| 🚶 离开 | 0.3 | ⏳ 再等等 |
| 🆘 需要关心 | 0.9 | ✅ 主动关心！ |

---

> 🧠💫 **想要更深？** 让 AI 真正*记住*、*发展关系*、拥有*情绪状态*。
>
> **→ [affective-longing](https://github.com/pearthink123/affective-longing)** — 记忆触发 + 关系状态机 + AI 自我情绪（VAD 模型）

---

## Use Any AI Backend

```python
# OpenAI / GPT
from revive_companion.adapters import OpenAIAdapter
adapter = OpenAIAdapter(config, api_key="sk-...")

# Anthropic / Claude
from revive_companion.adapters import AnthropicAdapter
adapter = AnthropicAdapter(config, api_key="sk-ant-...")

# Ollama / 本地模型
from revive_companion.adapters import GenericAdapter
adapter = GenericAdapter(config, api_url="http://localhost:11434/v1/chat/completions")
```

---

## How It Works

### The Math

Each tick, the engine computes hit probability:

```
P(hit) = 1 - e^(-λt)
```

Where λ = longing rate, t = time interval. Base: ~7.2% per 30-minute check.

### Probability Dynamics

| Event | Probability | Why |
|-------|------------|-----|
| Miss (no hit) | +8% | Longing builds |
| Hit → Hold | +8% | Longing suppressed |
| Hit → Send | Reset to 7.2% | Longing satisfied |

### The Curve

Over a night (midnight → 8am):
- 16 checks, all held
- Probability: 7% → 15% → 30% → 55% → 80% → 95%
- **This IS the longing — quantified, recorded, real**

---

## Architecture

```
revive-companion/
├── love.py              # Unified API (start here)
├── core/
│   ├── engine.py        # Poisson dice + probability dynamics
│   ├── config.py        # Pydantic v2 config with YAML support
│   └── models.py        # Data structures (Pydantic)
├── bayesian/
│   ├── core.py          # State estimation + send utility
│   └── learner.py       # Online learning from observations
├── info_gain/
│   ├── core.py          # Entropy × resolution potential
│   └── sources.py       # Silence, novelty, conversation state
├── control/
│   ├── pid.py           # PID controller (standalone use)
│   └── signal.py        # Pluggable signal framework
└── adapters/
    ├── openai.py        # OpenAI / GPT
    ├── anthropic.py     # Anthropic / Claude
    └── generic.py       # Ollama, HTTP, shell command
```

---

## Configuration

```yaml
engagement:
  lambda_rate: 0.15              # Base longing rate
  check_interval_minutes: 30     # Dice roll frequency
  growth_factor: 0.08            # How fast longing grows
  max_probability: 0.95          # Cap
  min_interval_hours: 1.0        # Anti-spam cooldown

  adjudication:
    quiet_hours:
      start: "00:00"
      end: "08:00"
    normal_send_probability: 0.7

persona:
  name: Companion
  tone: warm-brief
  context: "You are a caring companion checking in on your person."
```

---

## Dashboard

Visualize the AI engagement decision process: longing curves, state distributions, and send history.

```bash
pip install -e ".[dashboard]"
streamlit run dashboard.py
```

![Dashboard Overview](assets/dashboard-overview.png)
![Dashboard Detail](assets/dashboard-detail.png)

Features:
- 🎲 **Longing Curve** — Poisson probability over time
- 🧠 **State Distribution** — Bayesian-inferred user state
- ⏰ **Hourly Pattern** — When messages are most likely sent
- 📋 **Decision Log** — Detailed record of each decision

---

## Demos

```bash
git clone https://github.com/pearthink123/revive-companion
cd revive-companion
pip install -e .

PYTHONPATH=src python examples/quickstart.py              # Basic simulation
PYTHONPATH=src python examples/bayesian_demo.py           # State inference
PYTHONPATH=src python examples/bayesian_learning_demo.py  # Online learning
PYTHONPATH=src python examples/info_gain_demo.py          # Information gain
PYTHONPATH=src python examples/integration_example.py     # Smart notifier
```

---

## Integration

### Telegram Bot

```bash
pip install python-telegram-bot
python examples/telegram_bot.py --token YOUR_TOKEN --chat-id YOUR_CHAT_ID
```

### Discord / Slack / WeChat

Same pattern, just swap the send function:

```python
# Discord
await channel.send(message)

# Slack
slack_client.chat_postMessage(channel=channel_id, text=message)

# WeChat (itchat)
itchat.send(message, toUserName=friend_name)
```

---

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -v
```

124 tests covering:
- 🎲 Poisson engine (determinism, growth, timing)
- 🧠 Bayesian inference (state estimation, likelihood, learning)
- 📊 Information gain (decay, thresholds)
- 💘 Unified API (full pipeline)

---

## Consent & Safety

This library is designed for **respectful AI engagement**. Please use it responsibly:

### Built-in Protections
- **Quiet hours**: No messages during configured sleep periods
- **Minimum interval**: Anti-spam cooldown between messages
- **State inference**: Won't bother users who are busy or sleeping
- **Utility threshold**: Conservative default (0.5) — only sends when appropriate

### Best Practices
- ✅ **Opt-in**: Users should explicitly enable proactive messaging
- ✅ **Easy disable**: Users must be able to turn it off at any time
- ✅ **Transparency**: Users should know the AI can initiate contact
- ✅ **No emotional manipulation**: Don't use this to create dependency
- ❌ **No unsolicited contact**: Don't message users who didn't opt in
- ❌ **No persistence**: Respect when users want to be left alone

---

## What's Next

This is the **core engine** — focused on smart timing and decision-making.

For memory-triggered longing, relationship state machines, and AI self-emotion modeling:

**→ [affective-longing](https://github.com/pearthink123/affective-longing)** — Emotional extension (Memory + Relationship + Emotion)

---

## License

MIT
