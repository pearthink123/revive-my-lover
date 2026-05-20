"""
Telegram Bot Integration — revive-companion + Telegram

展示如何用 revive-companion 决定什么时候主动发消息给用户。

功能：
- 每 30 分钟检查一次
- 用 Poisson + Bayesian + InfoGain 决定是否发送
- 支持 quiet hours（不打扰）
- 记录用户回复，学习用户习惯

使用前：
1. 创建 Telegram bot（@BotFather）
2. 获取 bot token
3. 获取你的 chat_id（发消息给 @userinfobot）

运行：
    pip install python-telegram-bot
    python telegram_bot.py --token YOUR_TOKEN --chat-id YOUR_CHAT_ID
"""

import argparse
import asyncio
import logging
from datetime import datetime

# Telegram bot
try:
    from telegram import Bot
    from telegram.error import TelegramError
except ImportError:
    print("❌ 请安装: pip install python-telegram-bot")
    exit(1)

# revive-companion
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from revive_my_lover import PoissonLove
from revive_my_lover.bayesian import BayesianLearner

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramCompanion:
    """
    AI companion that decides when to message you via Telegram.

    Args:
        token: Telegram bot token.
        chat_id: Target chat ID.
        check_interval: Minutes between checks.
        quiet_hours: (start_hour, end_hour) when no messages are sent.
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        check_interval: int = 30,
        quiet_hours: tuple[int, int] = (0, 8),
    ):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self.check_interval = check_interval
        self.quiet_start, self.quiet_end = quiet_hours

        # Initialize revive-companion
        self.love = PoissonLove(seed=42)
        self.learner = BayesianLearner(min_observations=10)

        # State tracking
        self.last_user_reply: datetime | None = None
        self.last_message_sent: datetime | None = None
        self.message_count = 0

        logger.info("🤖 TelegramCompanion initialized")
        logger.info(f"   Chat ID: {chat_id}")
        logger.info(f"   Check interval: {check_interval} min")
        logger.info(f"   Quiet hours: {quiet_hours[0]:02d}:00 - {quiet_hours[1]:02d}:00")

    def is_quiet_hour(self, now: datetime | None = None) -> bool:
        """Check if current time is in quiet hours."""
        if now is None:
            now = datetime.now()
        hour = now.hour
        return self.quiet_start <= hour < self.quiet_end

    def calculate_reply_metrics(self) -> tuple[float, float]:
        """
        Calculate reply speed and length from last interaction.

        Returns:
            (reply_speed, reply_length) both in [0, 1].
        """
        if self.last_user_reply is None:
            return 0.5, 0.5  # Default

        now = datetime.now()
        hours_since = (now - self.last_user_reply).total_seconds() / 3600

        # Reply speed: 0 = very slow (>24h), 1 = instant (<5min)
        if hours_since < 0.083:  # 5 minutes
            reply_speed = 1.0
        elif hours_since < 1:
            reply_speed = 0.8
        elif hours_since < 4:
            reply_speed = 0.5
        elif hours_since < 12:
            reply_speed = 0.3
        else:
            reply_speed = 0.1

        # We don't have reply length, use default
        reply_length = 0.5

        return reply_speed, reply_length

    def decide(self) -> tuple[bool, str]:
        """
        Decide whether to send a message.

        Returns:
            (should_send, reason)
        """
        now = datetime.now()

        # Check quiet hours
        if self.is_quiet_hour(now):
            return False, f"Quiet hours ({self.quiet_start:02d}:00 - {self.quiet_end:02d}:00)"

        # Get reply metrics
        reply_speed, reply_length = self.calculate_reply_metrics()

        # Run the engine
        result = love.tick(now=now)

        # Update learner
        if result.user_state != "unknown":
            self.learner.record(
                state=result.user_state,
                reply_speed=reply_speed,
                reply_length=reply_length,
                hour=now.hour + now.minute / 60,
            )

        # Learn if ready
        if self.learner.should_update():
            params = self.learner.learn()
            love.update_params(params) if hasattr(love, "update_params") else None

        return result.should_send, result.reason

    async def send_message(self, text: str) -> bool:
        """
        Send a message via Telegram.

        Args:
            text: Message text.

        Returns:
            True if sent successfully.
        """
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode="Markdown")
            self.last_message_sent = datetime.now()
            self.message_count += 1
            logger.info(f"📤 Sent message #{self.message_count}: {text[:50]}...")
            return True
        except TelegramError as e:
            logger.error(f"❌ Failed to send: {e}")
            return False

    def record_user_reply(self):
        """Record that user replied (call this when you receive a message)."""
        self.last_user_reply = datetime.now()
        self.love.record_reply(reply_speed=0.8, reply_length=0.5)
        logger.info("📥 User reply recorded")

    async def run(self):
        """Main loop: check periodically and send if appropriate."""
        logger.info("🚀 Starting companion loop...")

        while True:
            try:
                now = datetime.now()
                logger.info(f"⏰ Check at {now.strftime('%H:%M')}")

                # Decide
                should_send, reason = self.decide()

                if should_send:
                    # Generate message (you can customize this)
                    messages = [
                        "💭 刚刚想到你了~",
                        "☀️ 今天过得怎么样？",
                        "🍵 记得喝水哦~",
                        "📚 学习/工作辛苦了！",
                        "🌙 很晚了，早点休息~",
                        "💪 加油！你可以的！",
                    ]
                    import random

                    msg = random.choice(messages)

                    await self.send_message(msg)
                    self.love.record_send()
                else:
                    logger.info(f"⏸️ Skip: {reason}")

                # Wait for next check
                await asyncio.sleep(self.check_interval * 60)

            except KeyboardInterrupt:
                logger.info("👋 Stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                await asyncio.sleep(60)  # Wait a minute on error


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="💘 revive-companion Telegram Bot")
    parser.add_argument("--token", required=True, help="Telegram bot token from @BotFather")
    parser.add_argument("--chat-id", required=True, help="Target chat ID (get from @userinfobot)")
    parser.add_argument(
        "--interval", type=int, default=30, help="Check interval in minutes (default: 30)"
    )
    parser.add_argument("--quiet-start", type=int, default=0, help="Quiet hours start (default: 0)")
    parser.add_argument("--quiet-end", type=int, default=8, help="Quiet hours end (default: 8)")

    args = parser.parse_args()

    # Create companion
    companion = TelegramCompanion(
        token=args.token,
        chat_id=args.chat_id,
        check_interval=args.interval,
        quiet_hours=(args.quiet_start, args.quiet_end),
    )

    # Run
    asyncio.run(companion.run())


if __name__ == "__main__":
    main()
