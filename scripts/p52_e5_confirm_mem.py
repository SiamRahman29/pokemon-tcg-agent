"""Short memory smoke for E5 confirm settings after sequencer release fix."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    path = str(ROOT / sub) if sub != "." else str(ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)

from ptcg.env import harness, sdk  # noqa: E402

import arena  # noqa: E402


def main() -> int:
    sdk.load()
    from sa import bcagent

    _, deck = arena.resolve_deck("grimmsnarl")
    _, treat = arena.build_agent(
        "bc:e5-confirm-mem,net=out/policy_v5.npz,seq,reply,sk8,sd32,sb8.0",
        deck,
    )
    _, ctrl = arena.build_agent("bc:e5-control,net=out/policy_v5.npz", deck)
    proc = psutil.Process(os.getpid())
    peaks: list[float] = []

    def on_game(match, seat, result):
        rss = proc.memory_info().rss / 1024 / 1024
        peaks.append(rss)
        print(
            f"match {match} seat{seat} rss_mb={rss:.0f} "
            f"turns={result.turns} planned={treat.seq.stats['planned']}",
            flush=True,
        )

    bcagent.reset_stats()
    t0 = time.time()
    harness.evaluate_paired(treat, ctrl, deck, deck, matches=3, on_game=on_game)
    print(
        "E5_MEM_OK",
        f"peak_mb={max(peaks):.0f}",
        f"final_mb={peaks[-1]:.0f}",
        f"planned={treat.seq.stats['planned']}",
        f"fallbacks={bcagent.STATS['fallbacks']}",
        f"aborts={treat.seq.stats['aborted_budget']}",
        f"sec={time.time() - t0:.1f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
