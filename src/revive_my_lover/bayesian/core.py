"""
Bayesian State Estimation — infer user's hidden state from observations.

Instead of "engagement high → send more", we ask:
"What is the user probably doing right now?" → then decide accordingly.

Hidden states:
  CHATTING    — user is in active conversation
  IDLE_ONLINE — user is online but not chatting
  BUSY        — user is occupied (work, eating, etc.)
  SLEEPING    — user is asleep
  AWAY        — user is away (no recent activity)
  NEEDING     — user might need a check-in (long silence + uncertain)

Observations:
  reply_speed  — how fast user replied (0-1)
  reply_length — how long the reply was (0-1)
  time_of_day  — hour (0-24)
  silence_hours — hours since last interaction
  has_reaction — did user react to last message

Usage:
    est = StateEstimator()
    est.update(reply_speed=0.8, reply_length=0.6, hour=14, silence=0.5)
    state, probs = est.most_likely()
    # → state=CHATTING, probs={CHATTING: 0.6, IDLE: 0.2, ...}
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(Enum):
    """Hidden user states."""
    CHATTING = "chatting"       # Active conversation
    IDLE_ONLINE = "idle"        # Online but quiet
    BUSY = "busy"               # Occupied
    SLEEPING = "sleeping"       # Asleep
    AWAY = "away"               # Not around
    NEEDING = "needing"         # Might need a check-in


# State transition probabilities: P(next | current)
# Row = current state, Col = next state
TRANSITIONS = {
    State.CHATTING: {
        State.CHATTING: 0.5, State.IDLE_ONLINE: 0.3,
        State.BUSY: 0.1, State.SLEEPING: 0.0, State.AWAY: 0.05, State.NEEDING: 0.05,
    },
    State.IDLE_ONLINE: {
        State.CHATTING: 0.15, State.IDLE_ONLINE: 0.4,
        State.BUSY: 0.2, State.SLEEPING: 0.05, State.AWAY: 0.1, State.NEEDING: 0.1,
    },
    State.BUSY: {
        State.CHATTING: 0.05, State.IDLE_ONLINE: 0.2,
        State.BUSY: 0.5, State.SLEEPING: 0.1, State.AWAY: 0.1, State.NEEDING: 0.05,
    },
    State.SLEEPING: {
        State.CHATTING: 0.02, State.IDLE_ONLINE: 0.1,
        State.BUSY: 0.08, State.SLEEPING: 0.7, State.AWAY: 0.05, State.NEEDING: 0.05,
    },
    State.AWAY: {
        State.CHATTING: 0.05, State.IDLE_ONLINE: 0.15,
        State.BUSY: 0.15, State.SLEEPING: 0.15, State.AWAY: 0.4, State.NEEDING: 0.1,
    },
    State.NEEDING: {
        State.CHATTING: 0.1, State.IDLE_ONLINE: 0.15,
        State.BUSY: 0.1, State.SLEEPING: 0.05, State.AWAY: 0.1, State.NEEDING: 0.5,
    },
}

# P(action | state) — what should we do in each state?
# Higher = more appropriate to send
SEND_UTILITY = {
    State.CHATTING:    0.2,  # Already chatting, don't interrupt
    State.IDLE_ONLINE: 0.7,  # Online but quiet, good time to reach out
    State.BUSY:        0.1,  # Busy, don't bother
    State.SLEEPING:    0.0,  # Sleeping, never send
    State.AWAY:        0.3,  # Away, maybe send (might see later)
    State.NEEDING:     0.9,  # Might need care, send!
}


@dataclass
class StateEstimator:
    """
    Bayesian state estimator.

    Maintains a probability distribution over hidden user states.
    Updates beliefs when new observations arrive.

    Attributes:
        prior: Initial state distribution.
        states: List of possible states.
    """

    prior: dict[State, float] = field(default_factory=lambda: {
        State.CHATTING: 0.1,
        State.IDLE_ONLINE: 0.2,
        State.BUSY: 0.3,
        State.SLEEPING: 0.1,
        State.AWAY: 0.2,
        State.NEEDING: 0.1,
    })

    def __post_init__(self):
        # Normalize prior
        total = sum(self.prior.values())
        if total > 0:
            self.prior = {k: v / total for k, v in self.prior.items()}
        self._belief = dict(self.prior)
        
        # Learned parameters (initially None)
        self._learned_likelihoods = None
        self._learned_temporal = None

    @property
    def belief(self) -> dict[State, float]:
        """Current belief distribution."""
        return dict(self._belief)

    def update(
        self,
        reply_speed: Optional[float] = None,
        reply_length: Optional[float] = None,
        hour: Optional[float] = None,
        silence_hours: Optional[float] = None,
        has_reaction: Optional[bool] = None,
    ) -> dict[State, float]:
        """
        Update beliefs with new observation.

        Bayesian update: P(state | obs) ∝ P(obs | state) × P(state)

        Only updates fields that are provided (None = no observation).
        """
        # Step 1: Transition (time passage → states shift naturally)
        self._transition()

        # Step 2: Observation likelihood
        likelihoods = {s: 1.0 for s in State}

        if reply_speed is not None:
            for s in State:
                likelihoods[s] *= self._likelihood_reply_speed(reply_speed, s)

        if reply_length is not None:
            for s in State:
                likelihoods[s] *= self._likelihood_reply_length(reply_length, s)

        if hour is not None:
            for s in State:
                likelihoods[s] *= self._likelihood_hour(hour, s)

        if silence_hours is not None:
            for s in State:
                likelihoods[s] *= self._likelihood_silence(silence_hours, s)

        if has_reaction is not None:
            for s in State:
                likelihoods[s] *= self._likelihood_reaction(has_reaction, s)

        # Step 3: Bayesian update
        unnormalized = {}
        for s in State:
            unnormalized[s] = likelihoods[s] * self._belief[s]

        # Normalize
        total = sum(unnormalized.values())
        if total > 0:
            self._belief = {k: v / total for k, v in unnormalized.items()}

        return self.belief

    def most_likely(self) -> tuple[State, dict[State, float]]:
        """Return the most likely state and full distribution."""
        best = max(self._belief, key=self._belief.get)
        return best, self.belief

    def send_utility(self) -> float:
        """
        Expected utility of sending a message.

        E[utility] = Σ P(state) × utility(send | state)
        """
        return sum(
            self._belief[s] * SEND_UTILITY[s]
            for s in State
        )

    def should_send(self, threshold: float = 0.5) -> tuple[bool, str]:
        """
        Decide whether to send based on expected utility.

        Returns (should_send, reason).
        """
        utility = self.send_utility()
        best_state, _ = self.most_likely()

        if utility >= threshold:
            return True, f"Utility {utility:.2f} ≥ {threshold} (likely: {best_state.value})"
        else:
            return False, f"Utility {utility:.2f} < {threshold} (likely: {best_state.value})"

    def reset(self, prior: Optional[dict[State, float]] = None) -> None:
        """Reset to initial beliefs."""
        self._belief = dict(prior or self.prior)

    def update_params(self, params: dict) -> None:
        """
        Update estimator parameters from learned data.
        
        Args:
            params: Dict from BayesianLearner.learn() containing:
                - transitions: P(next_state | current_state)
                - likelihoods: {state: {obs_key: (mean, std)}}
                - temporal: P(state | hour)
        """
        if "transitions" in params:
            # Update global TRANSITIONS (affects all instances)
            for from_state in State:
                if from_state in params["transitions"]:
                    for to_state in State:
                        if to_state in params["transitions"][from_state]:
                            TRANSITIONS[from_state][to_state] = params["transitions"][from_state][to_state]
        
        if "likelihoods" in params:
            # Store learned likelihoods for use in update()
            self._learned_likelihoods = params["likelihoods"]
        
        if "temporal" in params:
            # Store learned temporal patterns
            self._learned_temporal = params["temporal"]

    # ── Transition ──

    def _transition(self) -> None:
        """Apply state transition (beliefs shift naturally over time)."""
        new_belief = {s: 0.0 for s in State}
        for current in State:
            for next_s in State:
                new_belief[next_s] += self._belief[current] * TRANSITIONS[current][next_s]
        self._belief = new_belief

    # ── Likelihood functions ──

    def _likelihood_reply_speed(self, speed: float, state: State) -> float:
        """P(reply_speed | state)"""
        # Use learned parameters if available
        if (hasattr(self, '_learned_likelihoods') and 
            self._learned_likelihoods is not None and 
            state in self._learned_likelihoods):
            if "reply_speed" in self._learned_likelihoods[state]:
                mean, std = self._learned_likelihoods[state]["reply_speed"]
                return _gaussian(speed, mean, std)
        
        # Default profiles
        profiles = {
            State.CHATTING:    (0.8, 0.15),  # (mean, std)
            State.IDLE_ONLINE: (0.5, 0.2),
            State.BUSY:        (0.2, 0.15),
            State.SLEEPING:    (0.0, 0.05),
            State.AWAY:        (0.1, 0.1),
            State.NEEDING:     (0.3, 0.2),
        }
        mean, std = profiles[state]
        return _gaussian(speed, mean, std)

    def _likelihood_reply_length(self, length: float, state: State) -> float:
        """P(reply_length | state)"""
        # Use learned parameters if available
        if (hasattr(self, '_learned_likelihoods') and 
            self._learned_likelihoods is not None and 
            state in self._learned_likelihoods):
            if "reply_length" in self._learned_likelihoods[state]:
                mean, std = self._learned_likelihoods[state]["reply_length"]
                return _gaussian(length, mean, std)
        
        # Default profiles
        profiles = {
            State.CHATTING:    (0.7, 0.15),
            State.IDLE_ONLINE: (0.4, 0.2),
            State.BUSY:        (0.15, 0.1),
            State.SLEEPING:    (0.0, 0.05),
            State.AWAY:        (0.1, 0.1),
            State.NEEDING:     (0.3, 0.2),
        }
        mean, std = profiles[state]
        return _gaussian(length, mean, std)

    def _likelihood_hour(self, hour: float, state: State) -> float:
        """P(hour | state) — time of day makes some states more likely."""
        # Use learned temporal patterns if available
        if (hasattr(self, '_learned_temporal') and 
            self._learned_temporal is not None and 
            state in self._learned_temporal):
            hour_int = int(hour) % 24
            if hour_int in self._learned_temporal[state]:
                return self._learned_temporal[state][hour_int]
        
        # Default profiles
        profiles = {
            State.CHATTING:    [(9, 12), (17, 22)],  # Active hours
            State.IDLE_ONLINE: [(8, 12), (14, 18), (19, 23)],
            State.BUSY:        [(9, 12), (13, 17)],
            State.SLEEPING:    [(0, 7), (23, 24)],
            State.AWAY:        [(0, 24)],  # Any time
            State.NEEDING:     [(10, 12), (15, 18), (20, 23)],
        }
        for start, end in profiles[state]:
            if start <= hour < end:
                return 1.0
        return 0.1  # Low but not zero (night owl, etc.)

    def _likelihood_silence(self, hours: float, state: State) -> float:
        """P(silence_hours | state)"""
        # Use learned parameters if available
        if (hasattr(self, '_learned_likelihoods') and 
            self._learned_likelihoods is not None and 
            state in self._learned_likelihoods):
            if "silence_hours" in self._learned_likelihoods[state]:
                mean, std = self._learned_likelihoods[state]["silence_hours"]
                return _gaussian(hours, mean, std)
        
        # Default profiles
        profiles = {
            State.CHATTING:    (0.5, 1.0),   # (expected_hours, std)
            State.IDLE_ONLINE: (2.0, 2.0),
            State.BUSY:        (4.0, 3.0),
            State.SLEEPING:    (8.0, 3.0),
            State.AWAY:        (12.0, 6.0),
            State.NEEDING:     (24.0, 12.0),
        }
        mean, std = profiles[state]
        return _gaussian(hours, mean, std)

    def _likelihood_reaction(self, has_reaction: bool, state: State) -> float:
        """P(has_reaction | state)"""
        probs = {
            State.CHATTING:    0.7,
            State.IDLE_ONLINE: 0.4,
            State.BUSY:        0.1,
            State.SLEEPING:    0.0,
            State.AWAY:        0.05,
            State.NEEDING:     0.2,
        }
        p = probs[state]
        return p if has_reaction else (1 - p)


def _gaussian(x: float, mean: float, std: float) -> float:
    """Gaussian probability density (unnormalized)."""
    if std <= 0:
        return 1.0 if abs(x - mean) < 0.01 else 0.01
    return math.exp(-0.5 * ((x - mean) / std) ** 2)
