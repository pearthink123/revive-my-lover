# Poisson Longing 🎲

**Math models that make AI engagement feel human.**

![Poisson Longing Curve](assets/poisson_curve.png)

*Turn "thinking about you" into a measurable, adaptive curve.*

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

AI assistants are either **rigid** (fixed schedules) or **random** (no memory). Neither feels human.

## The Solution

Four math models working together:

| Stage | Model | Question | Answer |
|-------|-------|----------|--------|
| 🎲 **Timing** | Poisson process | When to consider? | Randomized like real "thinking of you" |
| 📊 **Value** | Information theory | Is this worth it? | Skip if you already know user's state |
| 🧠 **State** | Bayesian inference | What's user doing? | Infer hidden state → decide accordingly |

---

## Quick Start

```bash
pip install revive-my-lover
```

```python
from revive_my_lover import PoissonLove

love = PoissonLove()

result = love.tick()
if result.should_send:
    send_message(result.prompt)
    love.record_send()

# After user responds
love.record_reply(reply_speed=0.8, reply_length=0.6)
```

---

## How It Decides

The engine infers the user's **hidden state** from observations:

| Inferred State | Utility | Decision |
|---------------|---------|----------|
| 🗣️ **Chatting** | 0.2 | ❌ Don't interrupt |
| 💻 **Idle online** | 0.7 | ✅ Good time to reach out |
| 💼 **Busy** | 0.1 | ❌ Don't bother |
| 😴 **Sleeping** | 0.0 | ❌ Never send |
| 🚶 **Away** | 0.3 | ⏳ Maybe later |
| 🆘 **Needing** | 0.9 | ✅ Check in! |

No more "engaged → send more". Now it's: "user is probably busy → don't bother" or "user might need care → reach out".

---

## Use Any AI Backend

```python
# OpenAI / GPT
from revive_my_lover.adapters import OpenAIAdapter
adapter = OpenAIAdapter(config, api_key="sk-...")

# Anthropic / Claude
from revive_my_lover.adapters import AnthropicAdapter
adapter = AnthropicAdapter(config, api_key="sk-ant-...")

# Ollama / local models
from revive_my_lover.adapters import GenericAdapter
adapter = GenericAdapter(config, api_url="http://localhost:11434/v1/chat/completions")

# Run
from revive_my_lover.runner import Runner
runner = Runner(engine, adapter)
runner.run()
```

---

## Architecture

```
revive-my-lover/
├── love.py              # Unified API (start here)
├── core/
│   ├── engine.py        # Poisson dice + probability dynamics
│   ├── config.py        # YAML config
│   └── models.py        # Data structures
├── bayesian/
│   ├── core.py          # State estimation + send utility
│   └── __init__.py      # State enum, StateEstimator
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

## Demos

```bash
git clone https://github.com/pearthink123/revive-my-lover
cd revive-my-lover
pip install -e .

PYTHONPATH=src python examples/quickstart_unified.py    # 5-line quickstart
PYTHONPATH=src python examples/pid_demo.py              # PID controller
PYTHONPATH=src python examples/preference_demo.py       # 3 user styles
PYTHONPATH=src python examples/info_gain_demo.py        # Information gain
PYTHONPATH=src python examples/optimal_stop_demo.py     # Optimal stopping
PYTHONPATH=src python examples/full_pipeline_demo.py    # All 4 stages
```

---

## Why "Poisson"?

The Poisson process models events that happen independently at a constant average rate — like neurons firing, or "thinking about someone."

It's not random chaos. It's not rigid scheduling. It's **structured spontaneity** — the mathematical model of genuine, organic missing someone.

---

## License

MIT
