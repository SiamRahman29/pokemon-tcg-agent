"""Integrity smoke for E5's explicit-v5 turn sequencer.

Runs one paired mirror per scale point without archiving or reporting outcomes.
This is an execution and realized-compute check, not a strength screen.

    python -X utf8 scripts/p50_e5_smoke.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    path = str(ROOT / sub) if sub != "." else str(ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)

from ptcg.env import harness, sdk  # noqa: E402

import arena  # noqa: E402


NET = ROOT / "out/policy_v5.npz"
NET_SHA256 = "26c681c4845a7eb017def4ee5d353bbedd128767bfc034cb7091e95e0949849e"
ARMS = {
    "low": {"dets": 4, "budget_s": 1.0},
    "medium": {"dets": 8, "budget_s": 2.0},
    "high": {"dets": 16, "budget_s": 4.0},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not NET.exists():
        raise SystemExit(f"missing frozen baseline: {NET}")
    got = sha256(NET)
    if got != NET_SHA256:
        raise SystemExit(f"v5 hash changed: {got} != {NET_SHA256}")

    sdk.load()
    from sa import bcagent

    _, deck = arena.resolve_deck("grimmsnarl")
    realized: list[float] = []
    summaries: list[str] = []
    for name, config in ARMS.items():
        treatment_name, treatment = arena.build_agent(
            f"bc:e5-{name}-smoke,net=out/policy_v5.npz,seq,reply,"
            f"sk8,sd{config['dets']},sb{config['budget_s']}",
            deck,
        )
        _control_name, control = arena.build_agent(
            "bc:e5-control,net=out/policy_v5.npz",
            deck,
        )
        assert treatment.net is not None
        assert treatment.seq is not None
        assert treatment.seq.net is treatment.net, (
            "sequencer did not retain explicit net"
        )
        assert control.net is not None and control.seq is None

        bcagent.reset_stats()
        # Outcome is intentionally discarded: smoke games are not evidence.
        harness.evaluate_paired(treatment, control, deck, deck, matches=1)

        policy_stats = bcagent.STATS
        seq_stats = treatment.seq.stats
        assert policy_stats["fallbacks"] == 0, policy_stats
        assert policy_stats["net_missing"] == 0, policy_stats
        assert seq_stats["fellback"] == 0, seq_stats
        assert seq_stats["planned"] > 0, seq_stats
        assert seq_stats["sim_s"] > 0.0, seq_stats
        per_plan = seq_stats["sim_s"] / seq_stats["planned"]
        realized.append(per_plan)
        summaries.append(
            f"{name}:planned={seq_stats['planned']},"
            f"overruled={seq_stats['overruled']},"
            f"aborts={seq_stats['aborted_budget']},"
            f"s_per_plan={per_plan:.3f}"
        )
        assert treatment_name.startswith(f"bc:e5-{name}-smoke")

    assert realized[0] < realized[1] < realized[2], realized
    print("E5_SMOKE_OK " + " ".join(summaries) + " fallbacks=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
