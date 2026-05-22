"""
Optimal Stopping — demo scenarios.

Shows when to STOP (act now) vs CONTINUE (wait for better).

Run: PYTHONPATH=src python examples/optimal_stop_demo.py
"""

import random

from revive_companion.optimal_stop import (
    OptimalStop,
    SecretaryRule,
    ThresholdRule,
    UserActivitySignal,
)


def scenario_1():
    """Scenario 1: Threshold Rule — watching user activity over a day."""
    print("=== Scenario 1: Threshold Rule (观察用户活跃度) ===\n")
    print("  一天观察8次用户活跃度，阈值随时间降低（越来越急）\n")

    # Simulated user activity over 8 observations (8am to 10pm)
    activities = [0.2, 0.3, 0.5, 0.4, 0.7, 0.6, 0.85, 0.5]

    stop = OptimalStop(
        rule=ThresholdRule(
            horizon=8,
            value_range=(0, 1),
            urgency=0.5,
            observe_steps=2,  # First 2 just observe
        )
    )

    for t, act in enumerate(activities):
        result = stop.decide(act, step=t)
        print(f"  步骤{t + 1} | 活跃度={act:.2f} | 阈值={result.threshold:.2f} | {result.reason}")
        if result.should_stop:
            print(
                f"  → 🛑 在步骤{t + 1} 行动！（活跃度 {act:.2f} 超过阈值 {result.threshold:.2f}）"
            )
            break
    else:
        print("  → ⏳ 全部观察完，没有找到理想时机")


def scenario_2():
    """Scenario 2: Secretary Rule — classic 37% rule."""
    print("\n=== Scenario 2: Secretary Rule (37%法则) ===\n")
    print("  观察10个信号，前37%只观察，之后选第一个最好的\n")

    # Random signal values
    rng = random.Random(42)
    signals = [rng.random() for _ in range(10)]

    rule = SecretaryRule(horizon=10)
    stop = OptimalStop(rule=rule)

    for t, sig in enumerate(signals):
        result = stop.decide(sig, step=t)
        phase = "👁️观察" if t < 3 else "🎯决策"
        print(f"  步骤{t + 1} ({phase}) | 信号={sig:.3f} | {result.reason}")

        if result.should_stop:
            prev_best = max(stop.history[:-1]) if len(stop.history) > 1 else 0
            print(f"  → 🛑 在步骤{t + 1} 停止！（信号 {sig:.3f} 超过之前最佳 {prev_best:.3f}）")
            break
    else:
        print(f"  → 没有找到更好的，最后选了步骤{len(signals)}")


def scenario_3():
    """Scenario 3: Different urgency levels."""
    print("\n=== Scenario 3: 不同紧迫度对比 ===\n")

    signals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for urgency in [0.1, 0.5, 0.9]:
        stop = OptimalStop(
            rule=ThresholdRule(
                horizon=6,
                value_range=(0, 1),
                urgency=urgency,
            )
        )
        result = None
        for t, sig in enumerate(signals):
            result = stop.decide(sig, step=t)
            if result.should_stop:
                break

        if result and result.should_stop:
            print(
                f"  紧迫度={urgency:.1f} → 在步骤{result.step + 1}停止（信号={result.signal:.1f}，阈值={result.threshold:.2f}）"
            )
        else:
            print(f"  紧迫度={urgency:.1f} → 全部观察完未停止")


def scenario_4():
    """Scenario 4: Real engagement scenario."""
    print("\n=== Scenario 4: 实际场景 — 用户今天在线观察 ===\n")
    print("  每2小时观察一次用户状态，找最佳介入时机\n")

    # Simulate a day: user comes online at different times
    observations = [
        ("08:00", UserActivitySignal(hour=8, last_seen_minutes_ago=480, messages_today=0)),
        ("10:00", UserActivitySignal(hour=10, last_seen_minutes_ago=120, messages_today=1)),
        ("12:00", UserActivitySignal(hour=12, last_seen_minutes_ago=30, messages_today=3)),
        ("14:00", UserActivitySignal(hour=14, last_seen_minutes_ago=60, messages_today=3)),
        ("16:00", UserActivitySignal(hour=16, last_seen_minutes_ago=5, messages_today=5)),
        ("18:00", UserActivitySignal(hour=18, last_seen_minutes_ago=10, messages_today=7)),
        ("20:00", UserActivitySignal(hour=20, last_seen_minutes_ago=2, messages_today=8)),
    ]

    stop = OptimalStop(
        rule=ThresholdRule(
            horizon=len(observations),
            value_range=(0, 1),
            urgency=0.3,
            observe_steps=2,
        )
    )

    for t, (time_str, sig) in enumerate(observations):
        val = sig.value()
        result = stop.decide(val, step=t)
        print(
            f"  {time_str} | 活跃信号={val:.2f} | 阈值={result.threshold:.2f} | {'🛑行动' if result.should_stop else '⏳等待'}"
        )

        if result.should_stop:
            print(f"  → 最佳介入时机：{time_str}！")
            break


if __name__ == "__main__":
    print("=" * 60)
    print("Optimal Stop — 什么时候是最佳介入时机？")
    print("=" * 60)
    print()

    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()

    print()
    print("=" * 60)
    print("核心：观察信号，等阈值降低到信号以下时行动")
    print("  阈值高 → 只有超好的时机才行动（谨慎）")
    print("  阈值低 → 差不多就行（着急）")
    print("=" * 60)
