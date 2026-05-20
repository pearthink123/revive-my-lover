"""
PoissonLove — unified API.

Three core modules, one decision:
1. Poisson — when to consider
2. InfoGain — is it worth it?
3. Bayesian — what's the user doing?

Usage:
    from revive_my_lover import PoissonLove

    love = PoissonLove()
    result = love.tick()

    if result.should_send:
        send_message(result.prompt)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .core.engine import PoissonEngine
from .core.config import Config
from .core.models import Action, TickResult
from .control import Signal, CombinedSignal
from .info_gain import InformationGain, SilenceDuration, ConversationFlow, MessageNovelty
from .bayesian import StateEstimator, State


@dataclass
class LoveResult:
    """Result of a PoissonLove tick."""

    should_send: bool           # Final decision
    stage: str                  # Which stage decided ("poisson", "infogain", "bayesian")
    poisson_hit: bool           # Did Poisson dice hit?
    infogain_passed: bool       # Did InfoGain approve?
    user_state: str             # Inferred user state
    state_confidence: float     # Confidence in state inference
    send_utility: float         # Bayesian send utility (0-1)
    lambda_rate: float          # Current lambda rate
    probability: float          # Current Poisson probability
    info_gain: float            # Information gain value
    prompt: str                 # Message to send
    reason: str                 # Human-readable explanation
    metadata: dict = field(default_factory=dict)


class PoissonLove:
    """
    Probabilistic engagement engine for AI companions.

    Three core modules, one decision:
    1. Poisson process — randomized timing (when to consider)
    2. Information gain — is this interaction worth it?
    3. Bayesian inference — what is the user doing right now?

    Args:
        config: Engine configuration (or use defaults).
        infogain_threshold: Minimum info gain ratio to send.
        seed: Random seed for reproducibility.

    Example:
        >>> love = PoissonLove()
        >>> result = love.tick()
        >>> if result.should_send:
        ...     send_message(result.prompt)
        ...     love.record_send()
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        infogain_threshold: float = 0.20,
        seed: Optional[int] = None,
    ):
        # Config
        self.config = config or Config.from_dict({
            "engagement": {
                "lambda_rate": 0.15,
                "check_interval_minutes": 30,
                "growth_factor": 0.08,
                "max_probability": 0.95,
                "min_interval_hours": 1.0,
                "adjudication": {
                    "quiet_hours": {"start": "00:00", "end": "08:00"},
                    "normal_send_probability": 0.7,
                },
            },
            "persona": {
                "name": "Companion",
                "tone": "warm-brief",
                "context": "You are a caring companion.",
            },
        })

        # Modules
        self._engine = PoissonEngine(self.config, seed=seed)
        self._estimator = StateEstimator()
        self._infogain = InformationGain(threshold=infogain_threshold)

        # State
        self._base_lambda = self.config.engagement.lambda_rate
        self._last_user_reply: Optional[datetime] = None
        self._my_unanswered: int = 0
        self._recent_messages: list[str] = []
        self._last_reply_speed: float = 0.5
        self._last_reply_length: float = 0.5

    def tick(self, now: Optional[datetime] = None) -> LoveResult:
        """
        Run one tick of the full pipeline.

        Call this every check_interval_minutes.

        Returns:
            LoveResult with send decision and context.
        """
        if now is None:
            now = datetime.now()

        hour = now.hour + now.minute / 60.0

        # ── Stage 1: Poisson Dice ──
        poisson_result = self._engine.tick(now)

        if poisson_result.action != Action.HIT_SEND:
            return LoveResult(
                should_send=False,
                stage="poisson",
                poisson_hit=False,
                infogain_passed=False,
                user_state="unknown",
                state_confidence=0,
                send_utility=0,
                lambda_rate=self._base_lambda,
                probability=poisson_result.probability,
                info_gain=0,
                prompt="",
                reason=f"Poisson: {poisson_result.action.value}",
            )

        # ── Stage 2: Information Gain ──
        silence_hours = 0.0
        if self._last_user_reply:
            silence_hours = (now - self._last_user_reply).total_seconds() / 3600

        silence_src = SilenceDuration(last_reply_time=self._last_user_reply, now=now)
        flow_src = ConversationFlow(
            my_unanswered_messages=self._my_unanswered,
            user_replied_in_last_hour=(silence_hours < 1.0),
        )
        novelty_src = MessageNovelty(
            recent_messages=self._recent_messages[-5:],
            current_message="",
        )

        self._infogain.sources = [silence_src, flow_src, novelty_src]
        self._infogain._send_count = self._my_unanswered
        ig_result = self._infogain.evaluate()

        if not ig_result.worth_sending:
            return LoveResult(
                should_send=False,
                stage="infogain",
                poisson_hit=True,
                infogain_passed=False,
                user_state="unknown",
                state_confidence=0,
                send_utility=0,
                lambda_rate=self._base_lambda,
                probability=poisson_result.probability,
                info_gain=ig_result.gain,
                prompt="",
                reason=f"InfoGain: {ig_result.reason}",
            )

        # ── Stage 3: Bayesian State Estimation ──
        self._estimator.update(
            reply_speed=self._last_reply_speed,
            reply_length=self._last_reply_length,
            hour=hour,
            silence_hours=silence_hours,
        )

        best_state, state_probs = self._estimator.most_likely()
        send_utility = self._estimator.send_utility()
        state_confidence = state_probs[best_state]

        should_send, decision_reason = self._estimator.should_send(threshold=0.45)

        if not should_send:
            return LoveResult(
                should_send=False,
                stage="bayesian",
                poisson_hit=True,
                infogain_passed=True,
                user_state=best_state.value,
                state_confidence=state_confidence,
                send_utility=send_utility,
                lambda_rate=self._base_lambda,
                probability=poisson_result.probability,
                info_gain=ig_result.gain,
                prompt="",
                reason=f"Bayesian: {decision_reason}",
            )

        # ── All stages passed → Send! ──
        self._engine.confirm_send(now)
        prompt = self._build_prompt(now, best_state.value, poisson_result.probability)

        return LoveResult(
            should_send=True,
            stage="full",
            poisson_hit=True,
            infogain_passed=True,
            user_state=best_state.value,
            state_confidence=state_confidence,
            send_utility=send_utility,
            lambda_rate=self._base_lambda,
            probability=poisson_result.probability,
            info_gain=ig_result.gain,
            prompt=prompt,
            reason=f"Send (state={best_state.value}, utility={send_utility:.2f})",
        )

    def record_reply(
        self,
        message: str = "",
        reply_speed: float = 0.5,
        reply_length: float = 0.5,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Record that the user replied.

        Args:
            message: The reply text (for novelty tracking).
            reply_speed: How fast they replied (0-1).
            reply_length: How long the reply was (0-1).
            now: Timestamp (for simulation/testing). Defaults to datetime.now().
        """
        if now is None:
            now = datetime.now()
        self._last_user_reply = now
        self._my_unanswered = 0
        self._last_reply_speed = reply_speed
        self._last_reply_length = reply_length
        self._infogain.on_receive()

        # Update Bayesian estimator with the reply observation
        hour = now.hour + now.minute / 60.0
        self._estimator.update(
            reply_speed=reply_speed,
            reply_length=reply_length,
            hour=hour,
            silence_hours=0.0,
        )

        if message:
            self._recent_messages.append(message)

    def record_send(self, message: str = "", now: Optional[datetime] = None) -> None:
        """Record that we sent a message."""
        self._my_unanswered += 1
        self._infogain.on_send()
        if message:
            self._recent_messages.append(message)

    def _build_prompt(self, now: datetime, user_state: str, probability: float) -> str:
        """Build the message prompt based on inferred user state."""
        hour = now.hour
        if 6 <= hour < 10:
            time_ctx = "morning"
        elif 11 <= hour < 14:
            time_ctx = "midday"
        elif 14 <= hour < 18:
            time_ctx = "afternoon"
        elif 18 <= hour < 22:
            time_ctx = "evening"
        else:
            time_ctx = "late night"

        return (
            f"[Poisson Longing] "
            f"Time: {now.strftime('%H:%M')} ({time_ctx}), "
            f"User state: {user_state}, "
            f"Longing: {probability:.0%}"
        )
