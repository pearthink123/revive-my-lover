"""
Quick Start — 5 lines to run revive-companion.

Usage:
    python quickstart.py                    # Simulation mode (48h)
    python quickstart.py --live             # Real mode (runs forever)
    python quickstart.py --adapter openai   # Use OpenAI adapter
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from revive_my_lover import Config, PoissonEngine
from revive_my_lover.runner import Runner


def main():
    # 1. Load config
    config = Config.from_yaml(Path(__file__).parent / "pcpx.yaml")

    # 2. Create engine
    engine = PoissonEngine(config, seed=42)

    # 3. (Optional) Create adapter — uncomment to connect to real AI
    # adapter = OpenAIAdapter(config, api_key="sk-...")
    # adapter = AnthropicAdapter(config, api_key="sk-ant-...")
    # adapter = GenericAdapter(config, api_url="http://localhost:11434/v1/chat/completions", model="llama3")
    adapter = None

    # 4. Create runner
    runner = Runner(engine, adapter)

    # 5. Run
    if "--live" in sys.argv:
        print("🔴 LIVE MODE — running every 30 minutes")
        print("   Press Ctrl+C to stop\n")
        runner.run()
    else:
        print("📊 SIMULATION MODE — 48 hours\n")
        runner.run_simulation(hours=48)

        # Summary
        sends = [e for e in engine.log if e.action.value == "send"]
        holds = [e for e in engine.log if e.action.value == "hold"]
        misses = [e for e in engine.log if e.action.value == "miss"]
        print(f"\n{'=' * 50}")
        print(f"Total checks: {len(engine.log)}")
        print(f"Sent: {len(sends)} | Held: {len(holds)} | Missed: {len(misses)}")
        print(f"Max longing: {max(e.probability for e in engine.log):.0%}")

        # Save log
        engine.save_log(Path(__file__).parent / "simulation_log.json")
        print("\nLog saved to simulation_log.json")

        # Export curve data
        curve = engine.get_curve()
        print(f"Curve points: {len(curve)}")


if __name__ == "__main__":
    main()
