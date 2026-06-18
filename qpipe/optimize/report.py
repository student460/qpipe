"""Markdown report generation for an optimization run."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(v, nd=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if v <= -1e5 else f"{v:.{nd}f}"
    return str(v)


def write_report(result: dict, path: str | Path) -> None:
    s, cfg = result["summary"], result["config"]
    lines = [
        f"# Optimization Report — {result['run_id']}",
        "",
        f"**Strategy:** `{cfg['strategy_path']}` on **{cfg['symbol']}.{cfg['venue']}** ({cfg['bar_spec']})",
        f"**Objective:** {cfg['objective']} | **Windows:** {result['settings']['n_windows']} x "
        f"{result['settings']['segment_months']}m IS/OOS | **Trials/window:** {result['settings']['n_trials']} | "
        f"**Data:** {result['settings']['data_start']} .. {result['settings']['data_end']}",
        "",
        "## Recommendation",
        "",
        f"**Parameters:** `{json.dumps(s['recommended_params'])}`",
        f"- Median OOS {cfg['objective']}: **{_fmt(s['recommended_median_oos'])}**",
        f"- OOS windows positive: **{s['recommended_pct_positive']:.0%}**",
        "",
        "## Overfitting check (IS vs OOS)",
        "",
        f"- Median IS {cfg['objective']}: {_fmt(s['median_is_objective'])}",
        f"- Median OOS {cfg['objective']}: {_fmt(s['median_oos_objective'])}",
        "",
        "A large IS->OOS drop means the optimizer is fitting noise. Expect OOS below IS;"
        " worry when OOS collapses toward zero or goes negative.",
        "",
        "## Candidate parameter sets (cross-validated on all OOS windows)",
        "",
        "| Params | Median OOS | IQR | % positive |",
        "|---|---|---|---|",
    ]
    for c in result["candidates"]:
        lines.append(
            f"| `{json.dumps(c['params'])}` | {_fmt(c['median'])} | {_fmt(c['iqr'])} | {c['pct_positive']:.0%} |"
        )
    lines += [
        "",
        "## Per-window results",
        "",
        "| IS window | OOS window | Best params | IS obj | OOS obj |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(result["window_results"], key=lambda x: x["window"]["is_start"]):
        w = r["window"]
        lines.append(
            f"| {w['is_start']} .. {w['is_end']} | {w['oos_start']} .. {w['oos_end']} "
            f"| `{json.dumps(r['best_params'])}` | {_fmt(r['is_objective'])} | {_fmt(r['oos_objective'])} |"
        )
    Path(path).write_text("\n".join(lines))
