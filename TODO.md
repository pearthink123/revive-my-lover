# revive-companion TODO

## 📌 当前状态
- ✅ v0.9.0 已发布到 GitHub
- ✅ 124 个测试全过
- ✅ 中英文 README
- ✅ 技术博客
- ✅ Dashboard 可视化
- ✅ 集成示例（Telegram bot）
- ⭐ GitHub Stars: 0

---

## 🔥 短期（1-2 周）

### 推广
- [ ] **Reddit 帖子**
  - r/Replika — "I built a math engine that decides when AI should text you"
  - r/MachineLearning — 技术帖，讲数学原理
  - r/LocalLLaMA — "Works with any LLM backend"
  - r/Python — 库发布帖

- [ ] **Twitter/X**
  - 发 demo 视频（30秒）
  - Thread 讲解数学原理
  - @一些 AI 领域 KOL

- [ ] **中文社区**
  - V2EX — 开发帖
  - 小红书 — demo 视频
  - B站 — 技术讲解视频

### 产品
- [ ] **demo 视频** — 录制 Dashboard 运行效果
- [ ] **GIF 动图** — 渴望曲线变化过程
- [ ] **截图** — 决策日志、状态分布

---

## 🎯 中期（1-2 月）

### 集成
- [ ] **LangChain 集成** — 做一个 LangChain Tool
- [ ] **CrewAI 集成** — proactive agent 示例
- [ ] **AutoGen 集成** — multi-agent 场景
- [ ] **Discord bot** — 完整示例
- [ ] **微信机器人** — itchat/wechaty 示例

### 功能
- [ ] **PyPI 发布** — 正式上架
- [ ] **多用户支持** — 一个引擎服务多个用户
- [ ] **持久化** — 状态保存到数据库
- [ ] **Webhook 支持** — 事件驱动触发
- [ ] **Metrics API** — 暴露 Prometheus 指标

### 文档
- [ ] **API 文档** — 用 Sphinx/MkDocs 生成
- [ ] **使用案例** — 收集真实用户故事
- [ ] **对比分析** — vs cron, vs random, vs 固定策略

---

## 🚀 长期（3-6 月）

### 学术
- [ ] **arXiv 论文** — "Math-driven Proactive AI Engagement"
- [ ] **实验数据** — A/B test 结果
- [ ] **引用** — 找相关论文引用

### 商业化
- [ ] **高级 Dashboard** — 更多可视化
- [ ] **SaaS 版本** — 托管服务
- [ ] **企业功能** — 多租户、权限管理
- [ ] **SDK** — 移动端 SDK（iOS/Android）

### 社区
- [ ] **贡献指南** — CONTRIBUTING.md
- [ ] **Issue 模板** — Bug 报告、功能请求
- [ ] **Discord 社区** — 用户交流
- [ ] **每月更新** — Newsletter

---

## 🐛 已知问题

- [ ] OpenAIAdapter.send() 捕获所有异常后直接 return None，调试时会很痛
- [ ] record_reply() 强绑定 datetime.now()，不好做时间模拟
- [ ] load_log() 在 HIT_SEND 后恢复概率可能不符合发送后 reset 的语义

---

## 💡 想法池

- [ ] **情绪识别** — 从用户消息推断情绪状态
- [ ] **多模态** — 结合语音/图像/位置
- [ ] **A/B 测试框架** — 内置实验支持
- [ ] **可视化编辑器** — 拖拽配置规则
- [ ] **移动端** — Flutter/React Native SDK

---

## 📊 成功指标

| 指标 | 当前 | 目标（1月） | 目标（3月） |
|------|------|-----------|-----------|
| GitHub Stars | 0 | 10 | 100 |
| PyPI 下载 | 0 | 50 | 500 |
| Issues/PRs | 0 | 2 | 10 |
| 集成示例 | 2 | 5 | 10 |
| 用户故事 | 0 | 1 | 5 |

---

*最后更新: 2026-05-20*
