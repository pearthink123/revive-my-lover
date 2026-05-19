"""
PID Controller — example usage.

Shows how to use PIDController with custom signals
in different scenarios. Run: python examples/pid_demo.py
"""

from revive_my_lover.control import PIDController, Signal, CombinedSignal, ConstantSignal


# ─── Scenario 1: Chat engagement frequency ───

class ReplySpeedSignal(Signal):
    """Simulate reply speed: 0 (slow) to 1 (instant)."""

    def __init__(self):
        self._values = [0.2, 0.4, 0.8, 0.9, 0.3, 0.1, 0.5]
        self._i = 0

    def measure(self) -> float:
        v = self._values[self._i % len(self._values)]
        self._i += 1
        return v


class ReplyLengthSignal(Signal):
    """Simulate reply quality: short (0) to detailed (1)."""

    def __init__(self):
        self._values = [0.1, 0.3, 0.7, 0.9, 0.2, 0.1, 0.4]
        self._i = 0

    def measure(self) -> float:
        v = self._values[self._i % len(self._values)]
        self._i += 1
        return v


def demo_chat():
    print("=== Scenario 1: Chat Engagement ===")
    print("Target engagement score: 0.5")
    print("If user is engaged (high score) → decrease frequency")
    print("If user is disengaged (low score) → increase frequency\n")

    pid = PIDController(kp=0.2, ki=0.05, kd=0.1, setpoint=0.5)
    signals = CombinedSignal(
        (ReplySpeedSignal(), 0.6),
        (ReplyLengthSignal(), 0.4),
    )

    base_lambda = 0.15
    print(f"{'Tick':<5} {'Score':<8} {'Error':<8} {'Adj':<8} {'New λ':<8}")
    print("-" * 40)

    for i in range(7):
        score = signals.measure()
        adj = pid.update(score)
        new_lambda = max(0.05, min(0.4, base_lambda + adj))
        print(f"{i+1:<5} {score:<8.3f} {pid.error:<8.3f} {adj:<+8.3f} {new_lambda:<8.3f}")
        base_lambda = new_lambda


# ─── Scenario 2: Thermostat (classic PID) ───

def demo_thermostat():
    print("\n=== Scenario 2: Thermostat ===")
    print("Target: 22°C, current: 18°C\n")

    pid = PIDController(kp=2.0, ki=0.1, kd=0.5, setpoint=22.0)
    temp = 18.0

    print(f"{'Tick':<5} {'Temp':<8} {'Heat':<8}")
    print("-" * 25)

    for i in range(8):
        heat = pid.update(temp)
        temp += heat * 0.3  # simplified physics
        print(f"{i+1:<5} {temp:<8.1f} {heat:<+8.2f}")


# ─── Scenario 3: Game difficulty ───

def demo_game():
    print("\n=== Scenario 3: Game Difficulty ===")
    print("Target win rate: 50%\n")

    pid = PIDController(kp=0.5, ki=0.02, kd=0.1, setpoint=0.5)
    win_rates = [0.8, 0.7, 0.6, 0.55, 0.5, 0.48, 0.5]
    difficulty = 5.0  # 1-10 scale

    print(f"{'Tick':<5} {'Win%':<8} {'Adj':<8} {'Diff':<8}")
    print("-" * 32)

    for i, wr in enumerate(win_rates):
        adj = pid.update(wr)
        difficulty = max(1, min(10, difficulty - adj))
        print(f"{i+1:<5} {wr:<8.1%} {adj:<+8.2f} {difficulty:<8.1f}")


if __name__ == "__main__":
    demo_chat()
    demo_thermostat()
    demo_game()
