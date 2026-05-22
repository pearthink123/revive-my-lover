"""
Information Gain — demo scenarios.

Shows how entropy × resolution_potential drives send decisions.

Run: PYTHONPATH=src python examples/info_gain_demo.py
"""

from datetime import datetime, timedelta

from revive_companion.info_gain import (
    ConversationFlow,
    InformationGain,
    MessageNovelty,
    SilenceDuration,
    TimeOfDaySource,
)


def scenario_1():
    """Scenario 1: How long since last reply matters."""
    print("=== Scenario 1: Silence Duration ===\n")
    now = datetime(2026, 5, 19, 14, 0)

    for hours in [0.1, 0.5, 2, 8, 48]:
        src = SilenceDuration(last_reply_time=now - timedelta(hours=hours), now=now)
        gain = InformationGain(sources=[src], threshold=0.25)
        r = gain.evaluate()
        icon = "✅ SEND" if r.worth_sending else "❌ SKIP"
        print(
            f"  {hours:>5.1f}h silence | entropy={src.entropy():.2f} "
            f"resolution={src.resolution_potential():.2f} | "
            f"gain={r.gain:.3f} ({r.gain_ratio:.0%}) | {icon}"
        )


def scenario_2():
    """Scenario 2: Message novelty."""
    print("\n=== Scenario 2: Message Novelty ===\n")
    recent = ["你好呀", "在干嘛", "吃饭了吗"]

    src1 = MessageNovelty(recent_messages=recent, current_message="今天天气真好")
    r1 = InformationGain(sources=[src1], threshold=0.25).evaluate()
    print(
        f"  新消息  | resolution={src1.resolution_potential():.1f} | "
        f"gain={r1.gain:.3f} | {'✅' if r1.worth_sending else '❌'}"
    )

    src2 = MessageNovelty(recent_messages=recent, current_message="在干嘛")
    r2 = InformationGain(sources=[src2], threshold=0.25).evaluate()
    print(
        f"  重复消息 | resolution={src2.resolution_potential():.1f} | "
        f"gain={r2.gain:.3f} | {'✅' if r2.worth_sending else '❌'}"
    )


def scenario_3():
    """Scenario 3: Conversation flow."""
    print("\n=== Scenario 3: Conversation Flow ===\n")

    cases = [
        (
            "活跃对话 (用户刚回)",
            ConversationFlow(user_replied_in_last_hour=True, messages_in_last_hour=5),
        ),
        (
            "沉默 (啥也没发生)",
            ConversationFlow(user_replied_in_last_hour=False, messages_in_last_hour=0),
        ),
        ("已发1条没回", ConversationFlow(my_unanswered_messages=1)),
        ("已发3条没回", ConversationFlow(my_unanswered_messages=3)),
    ]
    for label, src in cases:
        r = InformationGain(sources=[src], threshold=0.25).evaluate()
        icon = "✅ SEND" if r.worth_sending else "❌ SKIP"
        print(
            f"  {label} | entropy={src.entropy():.1f} "
            f"resolution={src.resolution_potential():.1f} | "
            f"gain={r.gain:.3f} | {icon}"
        )


def scenario_4():
    """Scenario 4: Consecutive message decay."""
    print("\n=== Scenario 4: Consecutive Message Decay ===\n")
    src = SilenceDuration(
        last_reply_time=datetime(2026, 5, 19, 10, 0),
        now=datetime(2026, 5, 19, 14, 0),
    )
    gain = InformationGain(sources=[src], threshold=0.25, decay=0.7)
    for i in range(6):
        r = gain.evaluate()
        icon = "✅ SEND" if r.worth_sending else "❌ SKIP"
        print(f"  第{i + 1}条 | gain={r.gain:.3f} ({r.gain_ratio:.0%}) | {icon}  {r.reason}")
        gain.on_send()


def scenario_5():
    """Scenario 5: Combined sources — evening check-in."""
    print("\n=== Scenario 5: Combined Sources ===\n")
    sources = [
        SilenceDuration(
            last_reply_time=datetime(2026, 5, 19, 17, 0),
            now=datetime(2026, 5, 19, 20, 0),
        ),
        TimeOfDaySource(hour=20.0),
        MessageNovelty(
            recent_messages=["你好", "在干嘛"],
            current_message="今天看到一只超可爱的猫！",
        ),
    ]
    gain = InformationGain(sources=sources, threshold=0.25)
    r = gain.evaluate()
    print("  3h沉默 + 晚上8点 + 新话题")
    for s in sources:
        name = s.__class__.__name__
        print(f"    {name}: entropy={s.entropy():.2f} resolution={s.resolution_potential():.2f}")
    print(f"  总增益: {r.gain:.3f} bits ({r.gain_ratio:.0%})")
    print(f"  决策: {'✅ SEND' if r.worth_sending else '❌ SKIP'} — {r.reason}")


if __name__ == "__main__":
    print("=" * 60)
    print("Information Gain — Is this interaction worth it?")
    print("=" * 60)
    print()

    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()
    scenario_5()

    print()
    print("=" * 60)
    print("核心公式: gain = Σ(entropy × resolution_potential)")
    print("  高熵+高分辨率 = 发！（不确定，消息能获取信息）")
    print("  低熵+任意分辨率 = 别发（已经知道了）")
    print("  高熵+低分辨率 = 别发（不确定但消息也没用）")
    print("=" * 60)
