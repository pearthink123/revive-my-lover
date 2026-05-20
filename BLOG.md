# How I Made AI Engagement Feel Human — A Math-Driven Approach

*A probabilistic engine that decides when AI should reach out, not just what to say.*

---

## The Problem

Most AI companions have terrible timing. They either:

1. **Never initiate** — always waiting for you to start (boring)
2. **Spam you** — sending messages every hour regardless of context (annoying)
3. **Follow a schedule** — "Good morning!" at 9am sharp (robotic)

None of these feel human. When a real friend thinks of you, they don't follow a schedule. They think "I wonder how she's doing..." at random moments, and sometimes that thought grows into a message.

**How do we model that mathematically?**

---

## The Insight

Human engagement has three properties:

1. **Random timing** — We think of people at unpredictable moments
2. **Growing longing** — The longer we don't reach out, the more we want to
3. **Context-dependent** — We don't text someone who's obviously busy

These map to three mathematical models:

| Human Behavior | Math Model | Purpose |
|----------------|------------|---------|
| Random "thinking of you" moments | Poisson process | When to consider |
| Growing desire to reach out | Probability dynamics | How urgency builds |
| "Is now a good time?" | Bayesian inference | Whether to actually send |

---

## Module 1: Poisson Process (The Random Timing)

### The Math

A Poisson process models events that happen independently at a constant average rate. The probability of at least one event in time interval `t`:

```
P(≥1 event) = 1 - e^(-λt)
```

Where:
- `λ` = average rate (how often you "think of someone")
- `t` = time interval

### In Practice

Every 30 minutes, the engine rolls a dice:

```python
roll = random.random()
hit = roll < probability  # probability from Poisson formula
```

With `λ = 0.15` and 30-minute intervals, base probability is ~7.2%. Not too frequent, not too rare.

### Why This Works

- It's **memoryless** — each check is independent, like real "thinking of you" moments
- It's **not periodic** — unlike cron jobs, there's no fixed schedule
- It **feels organic** — sometimes you think of someone twice in an hour, sometimes not for days

---

## Module 2: Probability Dynamics (The Growing Longing)

### The Problem

A flat 7% probability doesn't capture how longing builds over time. If you haven't talked to someone in a week, you think of them more often.

### The Solution

Probability grows with each "miss" (when the dice doesn't hit):

```
If miss:
    P = min(P + growth_factor, max_probability)
If send:
    P = reset_to_base
```

### The Curve

Over a quiet night (midnight → 8am, no messages sent):

```
Time     Probability
00:00    7%
00:30    15%   (+8%)
01:00    23%   (+8%)
01:30    31%   (+8%)
...
07:30    95%   (capped)
```

**This IS the longing — quantified, recorded, real.**

By morning, the probability is 95%. The engine is practically begging to send a message. But should it?

---

## Module 3: Bayesian Inference (The Context Check)

### The Question

High probability ≠ should send. Even if you really want to reach out, you shouldn't text someone who's:

- In a meeting (busy)
- Asleep (sleeping)
- Already chatting with you (don't interrupt)

### The Math

We maintain a probability distribution over hidden user states:

```
P(state | observations) ∝ P(observations | state) × P(state)
```

We observe:
- **Reply speed** — how fast they replied last time (0-1)
- **Reply length** — how long their message was (0-1)
- **Time of day** — hour (0-24)
- **Silence duration** — hours since last interaction

### Hidden States

| State | Description | Send Utility |
|-------|-------------|--------------|
| 🗣️ Chatting | Active conversation | 0.2 (don't interrupt) |
| 💻 Idle online | Online but quiet | 0.7 (good time!) |
| 💼 Busy | Working, eating | 0.1 (don't bother) |
| 😴 Sleeping | Asleep | 0.0 (never send) |
| 🚶 Away | Not around | 0.3 (maybe later) |
| 🆘 Needing | Might need check-in | 0.9 (reach out!) |

### The Decision

Expected utility = Σ P(state) × utility(send | state)

If utility > threshold (default 0.5) → send

### Example

```
Observation: No reply for 48 hours, it's 8pm

P(CHAT)  = 0%    (no recent activity)
P(IDLE)  = 0%    (not recently online)
P(BUSY)  = 0%    (48h is too long for "busy")
P(SLEEP) = 0%    (it's 8pm)
P(AWAY)  = 0%    (48h suggests intentional silence)
P(NEED)  = 100%  (long silence + uncertain = might need care)

Utility = 1.0 × 0.9 = 0.9 → SEND! ✅
```

---

## Putting It Together

The full pipeline:

```
1. Poisson dice roll → Did we get a "thinking of you" moment?
   No → Skip
   Yes ↓

2. Information gain → Is this interaction worth it?
   No → Skip (we already know user's state)
   Yes ↓

3. Bayesian inference → What's the user doing?
   Low utility → Skip (they're busy/sleeping/chatting)
   High utility → SEND! ✅
```

Each module filters the decision:

| Stage | Catches | Example |
|-------|---------|---------|
| Poisson | Bad timing | "I just thought of you 5 minutes ago" |
| InfoGain | Redundant messages | "I already know she's at work" |
| Bayesian | Bad context | "She's probably sleeping at 3am" |

---

## Real-World Results

After running for 1 week with a real user:

- **Messages sent**: 12 (vs 48 with naive timer)
- **User response rate**: 83% (vs 23% with timer)
- **User sentiment**: "It feels like you actually think about me"

The key insight: **Less is more**. By being selective about timing, each message feels more meaningful.

---

## Online Learning

The engine learns from observations:

```python
learner = BayesianLearner()

# After each interaction
learner.record(
    state=inferred_state,
    reply_speed=0.8,
    hour=14.0,
)

# Periodically update
if learner.should_update():
    params = learner.learn()
    estimator.update_params(params)
```

After 100 observations, it learns:
- "User is usually BUSY from 9-17"
- "User replies fastest in the evening"
- "IDLE → BUSY is a common transition"

---

## Code Example

```python
from revive_my_lover import PoissonLove

love = PoissonLove()

# Every 30 minutes
result = love.tick()

if result.should_send:
    send_message("💭 Thinking of you~")
    love.record_send()

# When user replies
love.record_reply(reply_speed=0.8, reply_length=0.6)
```

Three lines. That's it.

---

## Beyond AI Companions

This isn't just for virtual girlfriends. Any "proactive AI" needs good timing:

- 🏥 **Health apps** — "Time to stretch?" (but not during a meeting)
- 📚 **Learning apps** — "Review this flashcard" (at optimal forgetting curve)
- 👴 **Elderly care** — "Checking in!" (when they might be lonely)
- 🛒 **Customer success** — "Need help?" (when they're stuck, not when they're browsing)

The math is the same. Only the message content changes.

---

## Try It Yourself

```bash
pip install git+https://github.com/pearthink123/revive-companion.git

# Run demo
git clone https://github.com/pearthink123/revive-companion
cd revive-companion
python examples/integration_example.py
```

GitHub: [pearthink123/revive-companion](https://github.com/pearthink123/revive-companion)

---

*Built with 💘 by someone who believes AI should know when to speak up — and when to shut up.*
