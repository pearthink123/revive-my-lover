"""
Bayesian Learning Demo — adapt to user patterns over time.

Shows how the BayesianLearner:
1. Records observations over time
2. Learns transition probabilities
3. Learns observation likelihoods
4. Learns temporal patterns
5. Updates the StateEstimator with learned parameters
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from revive_companion.bayesian import BayesianLearner, State, StateEstimator


def demo_learning():
    """Simulate learning from user observations."""
    print("=" * 60)
    print("🧠 Bayesian Learning Demo")
    print("=" * 60)
    print()
    print("Simulating a user's typical day:")
    print("  • Morning (7-9): Waking up, slow replies")
    print("  • Work hours (9-17): Busy, short replies")
    print("  • Evening (18-22): Free, fast replies")
    print("  • Night (22-7): Sleeping, no replies")
    print()

    # Create estimator and learner
    estimator = StateEstimator()
    learner = BayesianLearner(learning_rate=0.1, min_observations=10)

    # Simulate 100 observations
    print("📊 Recording 100 observations...")
    print()

    # Use a fixed seed for reproducibility
    import random

    rng = random.Random(42)

    for i in range(100):
        # Generate hour based on distribution
        hour = rng.randint(0, 23)

        # Generate state based on hour
        if 0 <= hour < 7:
            true_state = State.SLEEPING
            reply_speed = rng.gauss(0.0, 0.05)
            reply_length = rng.gauss(0.0, 0.05)
            silence_hours = rng.gauss(8.0, 2.0)
        elif 7 <= hour < 9:
            true_state = State.IDLE_ONLINE
            reply_speed = rng.gauss(0.4, 0.2)
            reply_length = rng.gauss(0.3, 0.15)
            silence_hours = rng.gauss(2.0, 1.0)
        elif 9 <= hour < 17:
            true_state = State.BUSY
            reply_speed = rng.gauss(0.2, 0.15)
            reply_length = rng.gauss(0.15, 0.1)
            silence_hours = rng.gauss(4.0, 2.0)
        elif 17 <= hour < 22:
            true_state = State.IDLE_ONLINE
            reply_speed = rng.gauss(0.7, 0.15)
            reply_length = rng.gauss(0.6, 0.2)
            silence_hours = rng.gauss(1.0, 0.5)
        else:
            true_state = State.IDLE_ONLINE
            reply_speed = rng.gauss(0.5, 0.2)
            reply_length = rng.gauss(0.4, 0.2)
            silence_hours = rng.gauss(2.0, 1.0)

        # Clamp values
        reply_speed = max(0.0, min(1.0, reply_speed))
        reply_length = max(0.0, min(1.0, reply_length))
        silence_hours = max(0.0, silence_hours)

        # Record observation
        learner.record(
            state=true_state,
            reply_speed=reply_speed,
            reply_length=reply_length,
            hour=hour,
            silence_hours=silence_hours,
        )

    print(f"✅ Recorded {learner.observation_count} observations")
    print()

    # Check if ready to learn
    if learner.should_update():
        print("📈 Learning parameters...")
        params = learner.learn()

        print()
        print("Learned transition probabilities:")
        print("-" * 40)
        for from_state in State:
            print(f"  {from_state.value:12} → ", end="")
            transitions = params["transitions"][from_state]
            # Sort by probability
            sorted_trans = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            for to_state, prob in sorted_trans[:3]:
                print(f"{to_state.value}: {prob:.1%}  ", end="")
            print()

        print()
        print("Learned likelihood parameters (mean, std):")
        print("-" * 40)
        for state in State:
            print(f"  {state.value:12}:")
            if state in params["likelihoods"]:
                for obs_key, (mean, std) in params["likelihoods"][state].items():
                    print(f"    {obs_key:15}: ({mean:.2f}, {std:.2f})")

        print()
        print("Confidence: {:.1%}".format(params["confidence"]))

        # Update estimator with learned parameters
        print()
        print("🔄 Updating estimator with learned parameters...")
        estimator.update_params(params)
        print("✅ Estimator updated!")

        # Get insights
        print()
        print("💡 Insights:")
        insights = learner.get_insights()
        print(f"  Most likely state: {insights['most_likely_state']}")
        print("  Peak hours per state:")
        for state, hours in insights["peak_hours"].items():
            if hours:
                print(f"    {state}: {', '.join(f'{h}:00' for h in hours)}")
        print("  Common transitions:")
        for pattern in insights["transition_patterns"]:
            print(f"    {pattern}")

    # Test the learned estimator
    print()
    print("=" * 60)
    print("🧪 Testing learned estimator")
    print("=" * 60)
    print()

    test_cases = [
        ("Morning, slow reply", 0.3, 0.2, 8.0, 2.0),
        ("Work hours, short reply", 0.15, 0.1, 14.0, 4.0),
        ("Evening, fast reply", 0.8, 0.7, 19.0, 0.5),
        ("Late night, no reply", 0.0, 0.0, 2.0, 8.0),
    ]

    for name, speed, length, hour, silence in test_cases:
        # Reset estimator
        estimator.reset()

        # Update with observation
        estimator.update(
            reply_speed=speed,
            reply_length=length,
            hour=hour,
            silence_hours=silence,
        )

        # Get most likely state
        state, probs = estimator.most_likely()
        utility = estimator.send_utility()
        should_send, reason = estimator.should_send(threshold=0.45)

        print(f"{name}:")
        print(f"  State: {state.value} ({probs[state]:.1%})")
        print(f"  Utility: {utility:.2f}")
        print(f"  Decision: {'SEND' if should_send else 'WAIT'}")
        print()


if __name__ == "__main__":
    demo_learning()
    print()
    print("=" * 60)
    print("✅ Bayesian learning demo completed!")
    print("=" * 60)
