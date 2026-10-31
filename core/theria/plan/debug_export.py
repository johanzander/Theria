"""The debug export: everything needed to understand what Theria did, in one markdown file.

Modelled on BESS Manager's `/api/export-debug-data`: auto-generated key findings first (what to
read), then the facts behind them: system, preflight, settings, the entities Theria reads and
writes, every recent plan run with its inputs (`scripts/replay_plan.py` plays one again), every
control cycle with what the pump and the thermostats showed and what was asked of them, the action
log, and today's log. Secrets are redacted wherever they sit.
"""

import json
import re
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

SECRET_KEY = re.compile(r"pass|token|secret|api_?key|authorization", re.IGNORECASE)
LOG_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.\d+)? \| (WARNING|ERROR|CRITICAL)\s*\| (.*)$"
)
KEY_LOG_PATTERN = re.compile(
    r"WARNING|ERROR|CRITICAL|failed|paused|wrote|LK thermostats|pump|plan computation|refit|mode",
    re.IGNORECASE,
)
LOG_TAIL = 200
MISMATCH_C = 0.1
MAX_LISTED = 12


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "***" if SECRET_KEY.search(str(k)) else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


def _label(hour: dict[str, Any]) -> str:
    return f"{hour['rts_c']:g}" + ("c" if hour.get("valves_closed") else "")


def first_hour_flips(runs: Sequence[dict[str, Any]]) -> list[tuple[datetime, str, str]]:
    """Where the first hour of one replan differs from the one before it (same hour): the pump
    would have been told something else 15 minutes later. 'c' marks closed floor loops.
    """
    flips = []
    for before, after in pairwise(runs):
        a, b = before["decided"][0], after["decided"][0]
        if a["start"] == b["start"] and _label(a) != _label(b):
            flips.append(
                (datetime.fromisoformat(after["computed_at"]), _label(a), _label(b))
            )
    return flips


def thermostat_mismatches(trace: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """A thermostat that shows something else than the cycle before asked of it."""
    found = []
    for before, after in pairwise(trace):
        asked = (before.get("lk") or {}).get("targets") or {}
        read = (after.get("lk") or {}).get("read") or {}
        for entity, target in asked.items():
            shown = read.get(entity)
            if shown is not None and abs(shown - target) > MISMATCH_C:
                found.append(
                    {
                        "at": after["at"],
                        "entity": entity,
                        "read": shown,
                        "asked": target,
                    }
                )
    return found


def log_anomalies(lines: Iterable[str]) -> list[dict[str, Any]]:
    """WARNING and above, the same message once with a count and when it first and last came."""
    seen: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for line in lines:
        match = LOG_LINE.match(line.strip())
        if match is None:
            continue
        stamp, level, rest = match.groups()
        message = rest.split(" - ", 1)[-1]
        entry = seen.setdefault(
            (level, message),
            {
                "level": level,
                "message": message,
                "count": 0,
                "first": stamp,
                "last": stamp,
            },
        )
        entry["count"] += 1
        entry["last"] = stamp
    return list(seen.values())


def key_findings(
    runs: Sequence[dict[str, Any]],
    trace: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    log_lines: Sequence[str],
    preflight: Sequence[tuple[str, str, str]],
) -> list[str]:
    out = []
    flips = first_hour_flips(runs)
    if flips:
        shown = ", ".join(f"{t:%H:%M} {a}->{b}" for t, a, b in flips[:MAX_LISTED])
        out.append(
            f"The first hour of the plan flipped {len(flips)} times between replans "
            f"(c = floor loops closed): {shown}"
        )
    mismatches = thermostat_mismatches(trace)
    for m in mismatches[:MAX_LISTED]:
        out.append(
            f"Thermostat {m['entity']} showed {m['read']} at {m['at'][11:16]} UTC; "
            f"{m['asked']} was asked in the cycle before"
        )
    problems = [e for e in events if e["action"] in ("failed", "paused", "reported")]
    for e in problems[-MAX_LISTED:]:
        out.append(f"{str(e['at'])[11:16]} {e['kind']} {e['action']}: {e['detail']}")
    for name, status, detail in preflight:
        if status != "ok":
            out.append(f"Preflight {status}: {name}: {detail}")
    for a in log_anomalies(log_lines)[:MAX_LISTED]:
        out.append(
            f"[log {a['level'].lower()}] x{a['count']} {a['message']} ({a['first']} -> {a['last']})"
        )
    return out


def read_log_lines(log_dir: str | None, days: Sequence[str]) -> list[str] | str:
    """The log files of `days` (oldest first), or a sentence about why there is none."""
    if log_dir is None:
        return "No log directory is configured."
    lines: list[str] = []
    for day in days:
        path = Path(log_dir) / f"theria-{day}.log"
        if path.exists():
            lines += path.read_text().splitlines()
    return lines or f"No log file for {', '.join(days)} in {log_dir}"


def compact_log(lines: list[str]) -> str:
    key = {i for i, line in enumerate(lines) if KEY_LOG_PATTERN.search(line)}
    tail = set(range(max(0, len(lines) - LOG_TAIL), len(lines)))
    out, previous = [], -1
    for i in sorted(key | tail):
        if previous >= 0 and i > previous + 1:
            out.append(f"[... {i - previous - 1} lines skipped ...]")
        out.append(lines[i])
        previous = i
    head = f"[Compact log: {len(key)} key lines of {len(lines)}, plus the last {LOG_TAIL}.]"
    return "\n".join([head, *out])


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _json(value: Any) -> str:
    return "```json\n" + json.dumps(redact(value), indent=2, default=_plain) + "\n```"


def build_report(
    *,
    now: datetime,
    version: str,
    mode: str,
    findings: list[str],
    preflight: Sequence[tuple[str, str, str]],
    settings: dict[str, Any],
    execution: dict[str, Any],
    entities: dict[str, Any],
    runs: Sequence[dict[str, Any]],
    trace: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    logs: list[str] | str,
    compact: bool = True,
) -> str:
    parts = [
        f"# Theria debug export\n\n**Exported**: {now.isoformat()}\n\n**Version**: {version}\n"
    ]

    parts.append("## Key findings (auto-generated, read first)\n")
    parts.append(
        "\n".join(f"- {f}" for f in findings)
        if findings
        else "No flips, thermostat mismatches, failures or log anomalies found."
    )

    parts.append(f"\n## System\n\n- Mode: {mode}\n- Time (UTC): {now.isoformat()}\n")

    parts.append("## Preflight\n")
    parts.append("\n".join(f"- {s.upper():4} {n}: {d}" for n, s, d in preflight))

    parts.append("\n## Settings\n")
    parts.append(_json(settings))
    parts.append("\n### Execution status\n")
    parts.append(_json(execution))

    parts.append(
        "\n## Entities\n\nWhat Home Assistant showed for the entities Theria reads and writes, at export time.\n"
    )
    parts.append(_json(entities))

    parts.append(
        "\n## Plan runs\n\nEvery plan computation with its inputs. Play one again with "
        "`python scripts/replay_plan.py runs.json --at <time UTC>`; save the array below (or fetch "
        "`/api/plan/runs?hours=N`) as runs.json.\n"
    )
    parts.append(
        "| computed (UTC) | mode | first hour | objective |\n|---|---|---|---|"
    )
    for r in runs:
        first = r["decided"][0] if r["decided"] else None
        parts.append(
            f"| {r['computed_at'][:19]} | {r['mode']} | "
            f"{'-' if first is None else _label(first)} | {r['objective_sek']:.2f} |"
        )
    parts.append("\n" + _json(list(runs) if not compact else list(runs)[-3:]))

    parts.append(
        "\n## Execution trace\n\nEvery control cycle: what the pump and the thermostats showed before it, what was asked, and what came of it.\n"
    )
    parts.append(
        "| at (UTC) | pump read | pump result | thermostats read | asked | result |\n|---|---|---|---|---|---|"
    )
    for t in trace[-96:]:
        lk = t.get("lk") or {}
        pump = t["pump"]
        parts.append(
            f"| {t['at'][:19]} | {pump['read']} | {pump['result']['action']} {pump['result']['target_c']} | "
            f"{lk.get('read')} | {lk.get('targets')} | {(lk.get('result') or {}).get('action')} |"
        )
    parts.append("\n" + _json(list(trace) if not compact else list(trace)[-12:]))

    parts.append("\n## Action log\n")
    parts.append(
        "\n".join(
            f"- {str(e['at'])[:19]} {e['kind']} {e['action']}: {e['detail']}"
            for e in events[-100:]
        )
    )

    parts.append("\n## System logs\n")
    parts.append(
        "```\n" + (compact_log(logs) if isinstance(logs, list) else logs) + "\n```"
    )
    return "\n".join(parts) + "\n"
