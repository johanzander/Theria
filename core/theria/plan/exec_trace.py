"""What each control cycle read, asked for and wrote, kept for a few days.

The plan runs (`plan_run.py`) keep what the planner was fed and decided. This is the other half: what
the pump and the thermostats showed before the cycle, what the cycle asked of them, what came of it,
and what the plan said for the hour. Only changes reach the action log; a thermostat that went back
to an old value between two cycles shows up here as two different readings.
"""

from datetime import datetime, timedelta

from core.theria.plan.recent_log import RecentLines

KEEP = timedelta(days=7)


class ExecTraceStore(RecentLines):
    def __init__(self, path: str, now: datetime):
        super().__init__(path, now, KEEP, "at")
