"""
Simple Integration Example — revive-companion + Any Bot

展示如何把 revive-companion 集成到任何 bot/agent 中。

核心逻辑：
1. 每隔一段时间检查
2. 调用 PoissonLove.tick() 决定是否发送
3. 如果发送，调用 record_send()
4. 收到回复后，调用 record_reply()

这个模式适用于：
- Telegram bot
- Discord bot
- Slack bot
- 微信机器人
- 手机通知
- 任何主动触达场景
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import datetime

from revive_companion import PoissonLove


class SmartNotifier:
    """
    智能通知器：决定什么时候该打扰用户。

    使用方式：
        notifier = SmartNotifier()

        # 每隔一段时间检查
        should_send, reason = notifier.check()

        if should_send:
            send_notification("想你了~")
            notifier.on_send()

        # 用户回复后
        notifier.on_reply()
    """

    def __init__(self, quiet_hours=(0, 8)):
        """
        Args:
            quiet_hours: (start, end) 不打扰的时间段，默认凌晨0-8点
        """
        self.love = PoissonLove()
        self.quiet_start, self.quiet_end = quiet_hours

    def check(self) -> tuple[bool, str]:
        """
        检查是否应该发送通知。

        Returns:
            (should_send, reason)
        """
        now = datetime.now()

        # 检查安静时间
        if self.quiet_start <= now.hour < self.quiet_end:
            return False, f"安静时间 ({self.quiet_start}:00-{self.quiet_end}:00)"

        # 调用引擎
        result = self.love.tick(now=now)

        return result.should_send, result.reason

    def on_send(self):
        """发送消息后调用。"""
        self.love.record_send()

    def on_reply(self, reply_speed=0.5, reply_length=0.5):
        """
        收到用户回复后调用。

        Args:
            reply_speed: 回复速度 (0-1)，1=秒回
            reply_length: 回复长度 (0-1)，1=很长
        """
        self.love.record_reply(
            reply_speed=reply_speed,
            reply_length=reply_length,
        )


# ═══════════════════════════════════════════════════════════
# 示例：模拟 24 小时
# ═══════════════════════════════════════════════════════════


def demo():
    """演示智能通知器的工作方式。"""
    print("=" * 60)
    print("💘 智能通知器演示")
    print("=" * 60)
    print()

    notifier = SmartNotifier(quiet_hours=(0, 8))

    # 模拟一天的检查
    print("时间   | 渴望度 | 状态     | 决策")
    print("-" * 50)

    for hour in range(24):
        for minute in [0, 30]:  # 每 30 分钟检查一次
            # 设置当前时间
            from datetime import datetime

            now = datetime(2026, 5, 20, hour, minute)

            # 检查
            result = notifier.love.tick(now=now)

            # 显示
            time_str = f"{hour:02d}:{minute:02d}"
            prob = f"{result.probability:.0%}"
            state = result.user_state[:8].ljust(8)

            if result.should_send:
                decision = "✅ 发送"
            elif notifier.quiet_start <= hour < notifier.quiet_end:
                decision = "🌙 安静"
            else:
                decision = "⏸️ 等待"

            # 每 2 小时显示一次
            if minute == 0 and hour % 2 == 0:
                print(f"{time_str} | {prob:6} | {state} | {decision}")

            # 模拟用户行为
            if 18 <= hour <= 22 and result.should_send:
                notifier.on_send()
                # 模拟用户回复
                notifier.on_reply(reply_speed=0.8, reply_length=0.6)

    print()
    print("=" * 60)
    print("✅ 演示完成")
    print()
    print("核心 API 只有 3 个方法：")
    print("  1. check()        → 检查是否应该发送")
    print("  2. on_send()      → 发送后调用")
    print("  3. on_reply()     → 收到回复后调用")
    print("=" * 60)


if __name__ == "__main__":
    demo()
