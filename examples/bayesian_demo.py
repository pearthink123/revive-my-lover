"""
Bayesian State Estimation — demo.

Shows how we infer user's hidden state from observations,
then decide whether to send.

Run: PYTHONPATH=src python examples/bayesian_demo.py
"""

from revive_my_lover.bayesian import StateEstimator, State


def print_beliefs(est, label=""):
    state, probs = est.most_likely()
    utility = est.send_utility()
    should, reason = est.should_send()

    print(f"  {label}")
    print(f"    Most likely: {state.value}")
    print(f"    Utility: {utility:.2f} → {'✅ SEND' if should else '❌ SKIP'} ({reason})")
    top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
    print(f"    Top states: {', '.join(f'{s.value}={p:.0%}' for s, p in top3)}")
    print()


def scenario_1():
    """Scenario 1: Morning — user just replied quickly."""
    print("=== 场景1: 早上用户快速回复 ===")
    est = StateEstimator()
    est.update(reply_speed=0.9, reply_length=0.7, hour=10, silence_hours=0.1)
    print_beliefs(est, "观测: 回复快 + 内容多 + 早上10点 + 刚聊过")


def scenario_2():
    """Scenario 2: Afternoon — no reply for 6 hours."""
    print("=== 场景2: 下午6小时没回复 ===")
    est = StateEstimator()
    est.update(hour=15, silence_hours=6)
    print_beliefs(est, "观测: 下午3点 + 6小时没回复")


def scenario_3():
    """Scenario 3: Night — silence for 2 days."""
    print("=== 场景3: 沉默2天 ===")
    est = StateEstimator()
    est.update(hour=20, silence_hours=48)
    print_beliefs(est, "观测: 晚上8点 + 48小时没回复")


def scenario_4():
    """Scenario 4: Active conversation — just replied with reaction."""
    print("=== 场景4: 正在聊天中 ===")
    est = StateEstimator()
    est.update(reply_speed=0.95, reply_length=0.8, hour=20, silence_hours=0.05, has_reaction=True)
    print_beliefs(est, "观测: 秒回 + 长消息 + 晚上 + 刚聊 + 有reaction")


def scenario_5():
    """Scenario 5: 3am — user is online but slow."""
    print("=== 场景5: 凌晨3点 ===")
    est = StateEstimator()
    est.update(reply_speed=0.2, hour=3, silence_hours=1)
    print_beliefs(est, "观测: 回复慢 + 凌晨3点 + 1小时没回复")


def scenario_6():
    """Scenario 6: Track beliefs over a conversation."""
    print("=== 场景6: 一场对话的信念变化 ===")
    est = StateEstimator()

    steps = [
        ("初始状态", {}),
        ("用户秒回", {"reply_speed": 0.9, "reply_length": 0.6, "hour": 14}),
        ("又秒回了", {"reply_speed": 0.85, "reply_length": 0.8, "hour": 14, "silence_hours": 0.05}),
        ("突然变慢", {"reply_speed": 0.2, "reply_length": 0.2, "hour": 14, "silence_hours": 0.5}),
        ("1小时没回", {"hour": 15, "silence_hours": 1.5}),
        ("3小时没回", {"hour": 17, "silence_hours": 3.5}),
        ("晚上8点还没回", {"hour": 20, "silence_hours": 6}),
    ]

    for label, obs in steps:
        if obs:
            est.update(**obs)
        state, probs = est.most_likely()
        utility = est.send_utility()
        should = "✅" if utility >= 0.5 else "❌"
        print(f"  {label:<18} → {state.value:<12} utility={utility:.2f} {should}")


if __name__ == "__main__":
    print("=" * 55)
    print("贝叶斯状态推断 — 用户在干嘛？该不该发？")
    print("=" * 55)
    print()

    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()
    scenario_5()
    scenario_6()

    print("=" * 55)
    print("核心：推断隐藏状态 → 计算发送效用 → 决策")
    print("  不是'积极就多发'，而是'ta在干嘛？该不该打扰？'")
    print("=" * 55)
