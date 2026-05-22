"""
Bayesian Learning — adapt to user patterns over time.

Learns from observations to improve:
1. Transition probabilities — how states change
2. Observation likelihoods — what observations mean
3. Temporal patterns — when states are likely

Usage:
    from revive_companion.bayesian import StateEstimator, BayesianLearner

    estimator = StateEstimator()
    learner = BayesianLearner()

    # After each observation
    learner.record(state, observation, timestamp)

    # Periodically update estimator
    if learner.should_update():
        new_params = learner.learn()
        estimator.update_params(new_params)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .core import TRANSITIONS, State


@dataclass
class ObservationRecord:
    """A single observation for learning."""

    state: State
    reply_speed: float | None = None
    reply_length: float | None = None
    hour: float | None = None
    silence_hours: float | None = None
    has_reaction: bool | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BayesianLearner:
    """
    Learns user patterns from observations.

    Tracks:
    1. State transitions — P(next_state | current_state)
    2. Observation likelihoods — mean/std for each state
    3. Temporal patterns — P(state | hour)

    Args:
        learning_rate: How fast to adapt (0-1). Higher = faster adaptation.
        min_observations: Minimum observations before learning starts.
        window_size: How many recent observations to consider.
    """

    learning_rate: float = 0.1
    min_observations: int = 20
    window_size: int = 100

    def __post_init__(self):
        # Transition counts: from_state -> to_state -> count
        self._transition_counts: dict[State, dict[State, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Observation values: state -> obs_key -> list of values
        self._observation_values: dict[State, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Temporal counts: state -> hour -> count
        self._temporal_counts: dict[State, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Recent observations for windowed learning
        self._recent: list[ObservationRecord] = []

        # State tracking
        self._last_state: State | None = None
        self._total_observations: int = 0

        # Learned parameters
        self._learned_transitions: dict[State, dict[State, float]] | None = None
        self._learned_likelihoods: dict[State, dict[str, tuple]] | None = None
        self._learned_temporal: dict[State, dict[int, float]] | None = None

    def record(
        self,
        state: State,
        reply_speed: float | None = None,
        reply_length: float | None = None,
        hour: float | None = None,
        silence_hours: float | None = None,
        has_reaction: bool | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Record an observation for learning.

        Args:
            state: Inferred user state.
            reply_speed: How fast user replied (0-1).
            reply_length: How long the reply was (0-1).
            hour: Hour of day (0-24).
            silence_hours: Hours since last interaction.
            has_reaction: Whether user reacted to last message.
            timestamp: When the observation occurred.
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Create record
        record = ObservationRecord(
            state=state,
            reply_speed=reply_speed,
            reply_length=reply_length,
            hour=hour,
            silence_hours=silence_hours,
            has_reaction=has_reaction,
            timestamp=timestamp,
        )

        # Update transition counts
        if self._last_state is not None:
            self._transition_counts[self._last_state][state] += 1

        # Update observation values
        if reply_speed is not None:
            self._observation_values[state]["reply_speed"].append(reply_speed)
        if reply_length is not None:
            self._observation_values[state]["reply_length"].append(reply_length)
        if silence_hours is not None:
            self._observation_values[state]["silence_hours"].append(silence_hours)

        # Update temporal counts
        if hour is not None:
            hour_int = int(hour) % 24
            self._temporal_counts[state][hour_int] += 1

        # Add to recent window
        self._recent.append(record)
        if len(self._recent) > self.window_size:
            self._recent.pop(0)

        # Update state tracking
        self._last_state = state
        self._total_observations += 1

    def should_update(self) -> bool:
        """Check if we have enough data to update parameters."""
        return self._total_observations >= self.min_observations

    def learn(self) -> dict:
        """
        Learn parameters from observations.

        Returns:
            Dict with learned parameters:
            - transitions: P(next_state | current_state)
            - likelihoods: {state: {obs_key: (mean, std)}}
            - temporal: P(state | hour)
        """
        result = {
            "transitions": self._learn_transitions(),
            "likelihoods": self._learn_likelihoods(),
            "temporal": self._learn_temporal(),
            "total_observations": self._total_observations,
            "confidence": self._calculate_confidence(),
        }

        return result

    def _learn_transitions(self) -> dict[State, dict[State, float]]:
        """Learn transition probabilities from observed data."""
        transitions = {}

        for from_state in State:
            transitions[from_state] = {}
            total = sum(self._transition_counts[from_state].values())

            if total > 0:
                for to_state in State:
                    count = self._transition_counts[from_state][to_state]
                    # Smooth with prior (add-1 smoothing)
                    transitions[from_state][to_state] = (count + 1) / (total + len(State))
            else:
                # No data, use default
                transitions[from_state] = dict(TRANSITIONS[from_state])

        self._learned_transitions = transitions
        return transitions

    def _learn_likelihoods(self) -> dict[State, dict[str, tuple]]:
        """Learn likelihood function parameters (mean, std)."""
        likelihoods = {}

        for state in State:
            likelihoods[state] = {}

            for obs_key in ["reply_speed", "reply_length", "silence_hours"]:
                values = self._observation_values[state][obs_key]

                if len(values) >= 5:  # Need at least 5 observations
                    mean = sum(values) / len(values)
                    # Calculate std
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std = math.sqrt(variance) if variance > 0 else 0.1

                    # Clamp std to reasonable range
                    std = max(0.05, min(0.5, std))

                    likelihoods[state][obs_key] = (mean, std)
                else:
                    # Not enough data, use default from core.py
                    if obs_key == "reply_speed":
                        profiles = {
                            State.CHATTING: (0.8, 0.15),
                            State.IDLE_ONLINE: (0.5, 0.2),
                            State.BUSY: (0.2, 0.15),
                            State.SLEEPING: (0.0, 0.05),
                            State.AWAY: (0.1, 0.1),
                            State.NEEDING: (0.3, 0.2),
                        }
                        likelihoods[state][obs_key] = profiles[state]
                    elif obs_key == "reply_length":
                        profiles = {
                            State.CHATTING: (0.7, 0.15),
                            State.IDLE_ONLINE: (0.4, 0.2),
                            State.BUSY: (0.15, 0.1),
                            State.SLEEPING: (0.0, 0.05),
                            State.AWAY: (0.1, 0.1),
                            State.NEEDING: (0.3, 0.2),
                        }
                        likelihoods[state][obs_key] = profiles[state]
                    elif obs_key == "silence_hours":
                        profiles = {
                            State.CHATTING: (0.5, 1.0),
                            State.IDLE_ONLINE: (2.0, 2.0),
                            State.BUSY: (4.0, 3.0),
                            State.SLEEPING: (8.0, 3.0),
                            State.AWAY: (12.0, 6.0),
                            State.NEEDING: (24.0, 12.0),
                        }
                        likelihoods[state][obs_key] = profiles[state]

        self._learned_likelihoods = likelihoods
        return likelihoods

    def _learn_temporal(self) -> dict[State, dict[int, float]]:
        """Learn temporal patterns: P(state | hour)."""
        temporal = {}

        # Count total per hour
        hour_totals: dict[int, int] = defaultdict(int)
        for state in State:
            for hour, count in self._temporal_counts[state].items():
                hour_totals[hour] += count

        # Calculate probabilities
        for state in State:
            temporal[state] = {}
            for hour in range(24):
                count = self._temporal_counts[state][hour]
                total = hour_totals[hour]

                if total > 0:
                    # Add-1 smoothing
                    temporal[state][hour] = (count + 1) / (total + len(State))
                else:
                    # No data, use uniform
                    temporal[state][hour] = 1.0 / len(State)

        self._learned_temporal = temporal
        return temporal

    def _calculate_confidence(self) -> float:
        """Calculate confidence in learned parameters (0-1)."""
        # Based on number of observations and coverage
        obs_factor = min(1.0, self._total_observations / 100)

        # Check transition coverage
        transition_coverage = 0
        for from_state in State:
            if sum(self._transition_counts[from_state].values()) > 0:
                transition_coverage += 1
        transition_factor = transition_coverage / len(State)

        # Check temporal coverage
        hours_with_data = set()
        for state in State:
            hours_with_data.update(self._temporal_counts[state].keys())
        temporal_factor = len(hours_with_data) / 24

        # Combined confidence
        confidence = (obs_factor + transition_factor + temporal_factor) / 3
        return confidence

    def get_insights(self) -> dict:
        """
        Get human-readable insights about learned patterns.

        Returns:
            Dict with insights:
            - most_likely_state: State with highest prior probability
            - peak_hours: Hours when each state is most likely
            - transition_patterns: Common state transitions
        """
        if not self.should_update():
            return {"status": "Learning not started yet", "observations": self._total_observations}

        insights = {
            "total_observations": self._total_observations,
            "confidence": self._calculate_confidence(),
            "most_likely_state": self._get_most_likely_state(),
            "peak_hours": self._get_peak_hours(),
            "transition_patterns": self._get_transition_patterns(),
        }

        return insights

    def _get_most_likely_state(self) -> str:
        """Find the most frequently observed state."""
        state_counts = defaultdict(int)
        for record in self._recent:
            state_counts[record.state] += 1

        if not state_counts:
            return "unknown"

        most_likely = max(state_counts, key=state_counts.get)
        return most_likely.value

    def _get_peak_hours(self) -> dict[str, list[int]]:
        """Find peak hours for each state."""
        peak_hours = {}

        for state in State:
            hours = []
            for hour in range(24):
                count = self._temporal_counts[state][hour]
                if count > 0:
                    hours.append((hour, count))

            # Sort by count, get top 3
            hours.sort(key=lambda x: x[1], reverse=True)
            peak_hours[state.value] = [h[0] for h in hours[:3]]

        return peak_hours

    def _get_transition_patterns(self) -> list[str]:
        """Find common transition patterns."""
        patterns = []

        for from_state in State:
            total = sum(self._transition_counts[from_state].values())
            if total == 0:
                continue

            # Find most common transition
            most_common = None
            max_count = 0
            for to_state, count in self._transition_counts[from_state].items():
                if count > max_count:
                    max_count = count
                    most_common = to_state

            if most_common and max_count > 0:
                probability = max_count / total
                if probability > 0.3:  # Only report significant patterns
                    patterns.append(f"{from_state.value} → {most_common.value} ({probability:.0%})")

        return patterns

    def reset(self) -> None:
        """Reset all learned data."""
        self._transition_counts.clear()
        self._observation_values.clear()
        self._temporal_counts.clear()
        self._recent.clear()
        self._last_state = None
        self._total_observations = 0
        self._learned_transitions = None
        self._learned_likelihoods = None
        self._learned_temporal = None

    @property
    def is_ready(self) -> bool:
        """Whether learner has enough data."""
        return self.should_update()

    @property
    def observation_count(self) -> int:
        """Total observations recorded."""
        return self._total_observations
