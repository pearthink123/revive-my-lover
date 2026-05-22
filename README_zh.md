# revive-companion 💘

**让 AI 主动互动更像人类的概率引擎。**

*不是"用户活跃就狂发"，而是"推断用户在干嘛，再决定要不要打扰"。*

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | 中文

---

## 问题

AI 助手要么**太被动**（等你来找它），要么**太刷屏**（定时轰炸），要么**太机械**（每天早上9点准时问好）。

这些都不像真人。

## 解决方案

三个数学模块，一个决策：

| 阶段 | 模块 | 问题 | 答案 |
|------|------|------|------|
| 🎲 **时机** | 泊松过程 | 什么时候考虑？ | 像真的"想你"一样随机 |
| 📊 **价值** | 信息论 | 值得发吗？ | 已经知道用户状态就别发了 |
| 🧠 **状态** | 贝叶斯推断 | 用户在干嘛？ | 推断隐藏状态 → 再决定 |

---

## 快速开始

```bash
pip install revive-companion
```

```python
from revive_companion import PoissonLove

love = PoissonLove()

# 每隔一段时间检查
result = love.tick()
if result.should_send:
    send_message("想你了~")
    love.record_send()

# 收到回复后
love.record_reply(reply_speed=0.8, reply_length=0.6)
```

**三行代码，搞定智能触达。**

---

## 它怎么决定？

引擎会推断用户的**隐藏状态**：

| 推断状态 | 发送效用 | 决策 |
|---------|---------|------|
| 🗣️ **聊天中** | 0.2 | ❌ 别打断 |
| 💻 **在线但安静** | 0.7 | ✅ 适合打招呼 |
| 💼 **在忙** | 0.1 | ❌ 别打扰 |
| 😴 **在睡觉** | 0.0 | ❌ 绝不发 |
| 🚶 **不在** | 0.3 | ⏳ 再等等 |
| 🆘 **可能需要关心** | 0.9 | ✅ 问候一下！ |

不是"活跃就多发"，而是"ta 在干嘛？该不该打扰？"

---

## 数学原理

### 泊松过程（随机时机）

每次检查，掷骰子：

```
P(命中) = 1 - e^(-λt)
```

- λ = 触达率（你"想某人"的频率）
- t = 时间间隔

默认参数：30 分钟检查一次，命中率 ~7.2%

### 渴望度曲线

如果没命中，概率会增长：

```
错过 → 概率 +8%
发送 → 概率重置
```

一晚上的曲线：
```
00:00  7%
01:00  23%
02:00  39%
...
07:00  95%  (好想发消息啊！)
```

**这就是"想念"——被量化、被记录、真实存在。**

### 贝叶斯推断（上下文判断）

```
P(状态|观测) ∝ P(观测|状态) × P(状态)
```

观测：
- 回复速度（秒回？还是很久才回？）
- 回复长度（长文？还是"嗯"？）
- 时间（凌晨3点？还是下午3点？）
- 沉默时长（刚聊完？还是两天没回了？）

---

## 集成示例

### Telegram Bot

```bash
pip install python-telegram-bot
python examples/telegram_bot.py --token YOUR_TOKEN --chat-id YOUR_CHAT_ID
```

### Discord / Slack / 微信

同样的模式，只需替换发送函数：

```python
# Discord
await channel.send(message)

# Slack
slack_client.chat_postMessage(channel=channel_id, text=message)

# 微信
itchat.send(message, toUserName=friend_name)
```

---

## Dashboard

可视化决策过程：渴望曲线、状态分布、发送历史。

```bash
pip install -e ".[dashboard]"
streamlit run dashboard.py
```

功能：
- 🎲 **渴望曲线** — Poisson 概率随时间变化
- 🧠 **状态分布** — Bayesian 推断的用户状态
- ⏰ **按小时分布** — 什么时候容易发消息
- 📋 **决策日志** — 每次决策的详细记录

---

## 在线学习

引擎会学习用户习惯：

```python
from revive_companion.bayesian import BayesianLearner

learner = BayesianLearner()

# 记录每次观察
learner.record(state=state, reply_speed=0.8, hour=14.0)

# 学习后更新
if learner.should_update():
    params = learner.learn()
    estimator.update_params(params)
```

100 次观察后，它会学到：
- "用户通常 9-17 点在忙"
- "晚上回复最快"
- "空闲→忙碌是常见转换"

---

## 适用场景

不只是 AI 伴侣！

| 场景 | 用法 |
|------|------|
| 🏥 健康助手 | 根据状态推断喝水/运动时机 |
| 📚 学习 buddy | 检测遗忘曲线，智能复习 |
| 👴 老人陪伴 | 检测"可能孤独"时温柔 check-in |
| 🛒 客服/电商 | 浏览后 proactive 跟进 |
| 🎮 NPC/游戏 | NPC 更真实地"想玩家" |
| 🏠 智能家居 | 根据作息推断开灯/关灯时机 |

**核心问题：什么时候该主动联系用户？**

---

## 测试

```bash
pip install -e ".[test]"
pytest tests/ -v
```

124 个测试覆盖：
- 🎲 Poisson 引擎（确定性、增长、时序）
- 🧠 贝叶斯推断（状态估计、似然、学习）
- 📊 信息增益（衰减、阈值）
- 💘 统一 API（完整管道）

---

## 安全与伦理

这个库为**尊重用户的 AI 互动**而设计。

### 内置保护
- **安静时间**：配置的睡眠期间不发消息
- **最小间隔**：消息间的反垃圾冷却
- **状态推断**：不会打扰忙碌或睡觉的用户
- **效用阈值**：保守默认值（0.5）——只有合适时才发送

### 最佳实践
- ✅ **用户主动开启**
- ✅ **随时可关闭**
- ✅ **透明**：用户知道 AI 可以主动联系
- ❌ **不制造情感依赖**
- ❌ **不骚扰未授权用户**

---

## 为什么叫 Poisson？

泊松过程模拟以恒定平均速率独立发生的事件——就像神经元放电，或者"想起某人"。

不是随机混乱，不是死板调度。是**结构化的自发性**——真正、有机的想念的数学模型。

---

## 许可证

MIT

---

## 链接

- 📦 [GitHub](https://github.com/pearthink123/revive-companion)
- 📖 [English README](README.md)
- 📝 [技术博客](BLOG.md)
