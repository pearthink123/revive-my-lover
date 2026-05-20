"""
Runner — the standalone scheduler that drives revive-my-lover.

Can run as:
  - A simple loop (blocking)
  - A one-shot tick (for integration into existing schedulers)
  - A background thread
"""

from __future__ import annotations
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

from .core.engine import PoissonEngine
from .core.config import Config
from .core.models import TickResult, Action
from .adapters.base import Adapter

logger = logging.getLogger("revive-my-lover")


class Runner:
    """
    Drives PoissonEngine or PoissonLove on a schedule.

    Usage (full pipeline):
        from revive_my_lover import PoissonLove
        love = PoissonLove()
        adapter = OpenAIAdapter(config, api_key="sk-...")
        runner = Runner(love, adapter)
        runner.run()

    Usage (timing only):
        engine = PoissonEngine(config)
        adapter = OpenAIAdapter(config, api_key="sk-...")
        runner = Runner(engine, adapter)
        runner.run()
    """

    def __init__(self, engine, adapter: Adapter = None,
                 on_tick: Callable[[TickResult], None] = None,
                 log_path: str | Path = None):
        # Support both PoissonEngine and PoissonLove
        self._love = None
        self._engine = None

        # Duck-type check for PoissonLove (has .tick() and ._engine)
        if hasattr(engine, '_engine') and hasattr(engine, '_estimator'):
            # PoissonLove instance
            self._love = engine
            self._engine = engine._engine
        else:
            self._engine = engine

        self.adapter = adapter
        self.on_tick = on_tick
        self.log_path = Path(log_path) if log_path else None
        self._running = False

    def tick(self, now: datetime = None) -> TickResult:
        """
        Run a single tick. Uses full PoissonLove pipeline if available,
        falls back to PoissonEngine timing only.

        Returns the TickResult. If result.should_send and adapter is set,
        also calls the AI and returns the response in result.metadata['response'].
        """
        if self._love:
            result = self._love.tick(now)
        else:
            result = self._engine.tick(now)

        if result.should_send and self.adapter:
            try:
                response = self.adapter.engage(result)
                result.metadata["response"] = response
                logger.info(f"AI responded: {response[:100] if response else 'None'}...")
            except Exception as e:
                logger.error(f"Adapter error: {e}")
                result.metadata["error"] = str(e)

        # Save log
        if self.log_path:
            self._engine.save_log(self.log_path)

        # Callback
        if self.on_tick:
            self.on_tick(result)

        return result

    def run(self, interval_minutes: int = None, max_ticks: int = None):
        """
        Blocking loop. Runs tick() every interval_minutes.

        Args:
            interval_minutes: Override check interval from config.
            max_ticks: Stop after N ticks (for testing). None = infinite.
        """
        if interval_minutes is None:
            interval_minutes = self._engine.config.engagement.check_interval_minutes

        self._running = True
        ticks = 0
        logger.info(f"Runner started. Interval: {interval_minutes}min")

        while self._running:
            result = self.tick()
            self._print_result(result)
            ticks += 1

            if max_ticks and ticks >= max_ticks:
                break

            time.sleep(interval_minutes * 60)

    def run_simulation(self, hours: int = 48, interval_minutes: int = None,
                       start: datetime = None):
        """
        Fast simulation — runs many ticks instantly for testing/visualization.

        Args:
            hours: How many hours to simulate.
            interval_minutes: Override check interval.
            start: Start time. Defaults to now.
        """
        if interval_minutes is None:
            interval_minutes = self._engine.config.engagement.check_interval_minutes
        if start is None:
            start = datetime.now()

        total_ticks = int(hours * 60 / interval_minutes)
        delta = timedelta(minutes=interval_minutes)

        for i in range(total_ticks):
            tick_time = start + delta * i
            result = self.tick(tick_time)

        logger.info(f"Simulation complete: {total_ticks} ticks over {hours}h")

    def stop(self):
        """Stop the blocking loop."""
        self._running = False

    def _print_result(self, result: TickResult):
        """Pretty-print a tick result."""
        h = int(result.hour_of_day)
        m = int((result.hour_of_day % 1) * 60)
        ts = f"{h:02d}:{m:02d}"
        prob = f"{result.probability:.0%}"

        if result.action == Action.SKIP:
            print(f"  [{ts}] SKIP  P={prob} | {result.reason}")
        elif result.action == Action.MISS:
            if result.probability > 0.5:
                print(f"  [{ts}] miss  P={prob} (high longing, no hit)")
        elif result.action == Action.HIT_SEND:
            print(f"  [{ts}] SEND  P={prob} | {result.reason}")
        elif result.action == Action.HIT_HOLD:
            print(f"  [{ts}] HOLD  P={prob} | {result.reason}")
