"""Track C's deck search: emit the pre-registered variants and the runs to make.

⛔ **The candidate list is FROZEN in `out/logs/deck_search_prereg.txt`, which was
committed (`d93cf04`) BEFORE this file or any variant deck existed.** Nothing
here may be edited to add, drop or retune a candidate after a result is seen;
that is the multiplicity the two-stage design exists to prevent (§8ao's beta
sweep is the precedent for declining it).

**What this does.** Writes each variant as a headerless 60-line deck CSV under
`out/decks/` -- `arena.resolve_deck` accepts a `.csv` path directly, so no
`decks/*.py` module is created per candidate and there is nothing to leave
behind. Then prints the exact `arena.py play` commands for stage 1.

**Why it prints commands instead of scoring.** Rule 18: `arena.py` already emits
the seat-corrected score, its Wilson CI and the per-seat W/D/L. Re-implementing
that here could only introduce error -- it is exactly how §8an nearly published
0.510 for 0.888.

    python -X utf8 scripts/p36_deck_search.py            # write CSVs + commands
    python -X utf8 scripts/p36_deck_search.py --rank     # rank stage-1 archives

Stage 1 is a RANKING, not a test. No p-value is computed here by design.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "decks", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from decks import grimmsnarl  # noqa: E402

NET = "out/policy_v5.npz"
DECK_DIR = ROOT / "out" / "decks"
ARENA_DIR = ROOT / "out" / "arena"

# --- the frozen list ---------------------------------------------------------
# (id, label, delta) -- delta is {card_id: count change}, always 1-for-1.
MIRROR = [
    ("A", "Dawn -> Ultra Ball",              {1231: -1, 1121: +1}),
    ("B", "Pokegear 3.0 -> Ultra Ball",      {1122: -1, 1121: +1}),
    ("C", "Unfair Stamp -> Ultra Ball",      {1080: -1, 1121: +1}),
    ("D", "Rare Candy 3->2 -> Ultra Ball",   {1079: -1, 1121: +1}),
    ("E", "Night Stretcher 3->2 -> Ultra Ball", {1097: -1, 1121: +1}),
    ("F", "Buddy-Buddy Poffin 4->3 -> Ultra Ball", {1086: -1, 1121: +1}),
    ("G", "Pokegear 3.0 -> Fezandipiti ex",  {1122: -1, 140: +1}),
    ("H", "Dawn -> Latias ex",               {1231: -1, 184: +1}),
]
STRAT = [
    ("I", "Tool Scrapper -> Ultra Ball",     {1137: -1, 1121: +1}),
    ("J", "Pokegear 3.0 -> Froslass 2->3",   {1122: -1, 104: +1}),
    ("K", "Dawn -> Snorunt 2->3",            {1231: -1, 860: +1}),
]
# §8ar: the two anchors where the mirror-blind cards are most live.
STRAT_ANCHORS = [("alakazam5", "rule:alakazam5", "alakazam5", 0.220),
                 ("crustle", "rule:crustle", "crustle_v1", 0.067)]
MIRROR_SHARE = 0.333
N_MIRROR = 4000
N_STRAT = 2000


def write_deck(name: str, delta: dict[int, int]) -> Path:
    counts = dict(grimmsnarl.DECKLIST)
    for cid, d in delta.items():
        counts[cid] = counts.get(cid, 0) + d
        if counts[cid] < 0:
            raise SystemExit(f"{name}: negative count for {cid}")
        if counts[cid] == 0:
            del counts[cid]
    total = sum(counts.values())
    if total != 60:
        raise SystemExit(f"{name}: deck is {total} cards, not 60")
    ids = [cid for cid, n in counts.items() for _ in range(n)]
    DECK_DIR.mkdir(parents=True, exist_ok=True)
    path = DECK_DIR / f"{name}.csv"
    path.write_text("\n".join(str(i) for i in ids), encoding="utf-8")
    return path


def read_score(tag: str) -> tuple[float, int, tuple[float, float]] | None:
    """Take the score from the line `arena.py` PRINTED. Do not recompute it.

    🔴 Rule 18, and this function is a direct consequence of nearly breaking it.
    The first version of this script scored the archive itself, deriving our
    seat from `deck0 == <variant name>`. That works for a variant (the two deck
    names differ) and **silently fails for the stock-vs-stock control**, where
    both sides are named `grimmsnarl`: every row reads as seat 0, so the
    "control" would have been player 0's win rate -- and §8aj measured a ~1 pp
    first-player advantage, so it would have come back near 0.510 and looked
    perfectly reasonable. A plausible number, not a crash, for the fifth time.

    `arena.py` already alternates seats and prints the seat-corrected score with
    its Wilson CI. Reading that line cannot make this mistake.
    """
    log = ROOT / "out" / "logs" / f"p36_{tag}.log"
    if not log.exists():
        return None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "score=" in line and "[" in line:
            try:
                sc = float(line.split("score=")[1].split()[0])
                lo, hi = line.split("[")[1].split("]")[0].split(",")
                n = int(line.split(" over ")[1].split()[0])
                return sc, n, (float(lo), float(hi))
            except (IndexError, ValueError):
                continue
    return None


def cmd_emit() -> int:
    print("\nWriting the 11 pre-registered variants to out/decks/ ...\n")
    for cid, label, delta in MIRROR + STRAT:
        p = write_deck(cid, delta)
        print(f"  {cid}  {label:<38} -> {p.relative_to(ROOT)}")

    print("\n" + "=" * 79)
    print("STEP 0 -- THE FALSIFICATION CHECK. Nothing else runs until this passes.")
    print("  Stock vs stock must reproduce §8aj's 0.4980 [0.483, 0.513].")
    print("=" * 79)
    print(f'python -X utf8 scripts/arena.py play "bc:v5,net={NET}" "bc:v5,net={NET}" '
          f'--deck-a grimmsnarl --deck-b grimmsnarl --matches {N_MIRROR // 2} '
          f'--archive out/arena/p36_ctrl_mirror.jsonl '
          f'> out/logs/p36_ctrl_mirror.log 2>&1')

    print("\n" + "=" * 79)
    print(f"STAGE 1a -- MIRROR SCREEN, n={N_MIRROR}, direct head-to-head.")
    print("  RANKING ONLY. No candidate is called a winner here.")
    print("=" * 79)
    for cid, label, _ in MIRROR:
        print(f'# {cid}: {label}')
        print(f'python -X utf8 scripts/arena.py play "bc:v5,net={NET}" "bc:v5,net={NET}" '
              f'--deck-a out/decks/{cid}.csv --deck-b grimmsnarl '
              f'--matches {N_MIRROR // 2} --archive out/arena/p36_{cid}.jsonl '
              f'> out/logs/p36_{cid}.log 2>&1')

    print("\n" + "=" * 79)
    print(f"STAGE 1b -- STRATIFIED SCREEN, n={N_STRAT} per arm, mirror-blind cards.")
    print("  The stock-60 control at each anchor is measured ONCE and shared.")
    print("=" * 79)
    for akey, aspec, adeck, _ in STRAT_ANCHORS:
        print(f'# shared control: stock 60 vs {aspec}')
        print(f'python -X utf8 scripts/arena.py play "bc:v5,net={NET}" "{aspec}" '
              f'--deck-a grimmsnarl --deck-b {adeck} --matches {N_STRAT // 2} '
              f'--archive out/arena/p36_ctrl_{akey}.jsonl '
              f'> out/logs/p36_ctrl_{akey}.log 2>&1')
    for cid, label, _ in STRAT:
        print(f'# {cid}: {label}')
        for akey, aspec, adeck, _ in STRAT_ANCHORS:
            print(f'python -X utf8 scripts/arena.py play "bc:v5,net={NET}" "{aspec}" '
                  f'--deck-a out/decks/{cid}.csv --deck-b {adeck} '
                  f'--matches {N_STRAT // 2} --archive out/arena/p36_{cid}_{akey}.jsonl '
                  f'> out/logs/p36_{cid}_{akey}.log 2>&1')

    tot = N_MIRROR * (len(MIRROR) + 1) + N_STRAT * (len(STRAT) + 1) * len(STRAT_ANCHORS)
    print(f"\n  total stage 1: {tot:,} games "
          f"(~{tot / 5.96 / 3600:.1f} h single-process, "
          f"~{tot / 5.96 / 3600 / 2.5:.1f} h at rule 7's 2-3 jobs)")
    return 0


def cmd_rank() -> int:
    print("\nSTEP 0 -- falsification check (stock vs stock):")
    got = read_score("ctrl_mirror")
    if got is None:
        print("  NOT RUN.  ⛔ Nothing may be ranked until it is.")
        return 1
    s, n, (lo, hi) = got
    ok = lo <= 0.4980 <= hi or (0.483 <= s <= 0.513)
    print(f"  {s:.4f} [{lo:.3f}, {hi:.3f}] over {n} games -- "
          f"{'✅ consistent with §8aj 0.4980' if ok else '🔴 DOES NOT REPRODUCE §8aj'}")
    if not ok:
        print("  ⛔ The harness has changed. Every number below it is suspect.")
        return 1

    print(f"\nSTAGE 1a -- MIRROR SCREEN (n={N_MIRROR}). Ranking statistic is the")
    print(f"  implied change in field-weighted W, = {MIRROR_SHARE:.3f} x (score - control).")
    print("  ⚠ These are NOT tests. The CIs are shown so the ranking's noise is")
    print("    visible, not so a candidate can be declared significant.\n")
    print("  %-3s %-38s %8s %18s %9s" % ("id", "swap", "score", "95% CI", "dW"))
    rows = []
    for cid, label, _ in MIRROR:
        got = read_score(cid)
        if got is None:
            print("  %-3s %-38s %8s" % (cid, label, "not run"))
            continue
        sc, nn, (l, h) = got
        dw = MIRROR_SHARE * (sc - s)
        rows.append((dw, cid, label, sc, l, h, nn))
        print("  %-3s %-38s %8.4f  [%.3f, %.3f] %9.5f"
              % (cid, label, sc, l, h, dw))

    print(f"\nSTAGE 1b -- STRATIFIED SCREEN (n={N_STRAT}/arm), mirror-blind cards.\n")
    print("  %-3s %-38s %9s %9s %9s" % ("id", "swap", "alakazam5", "crustle", "dW"))
    for cid, label, _ in STRAT:
        parts, dw = [], 0.0
        for akey, _, _, share in STRAT_ANCHORS:
            cg, vg = read_score(f"ctrl_{akey}"), read_score(f"{cid}_{akey}")
            if cg is None or vg is None:
                parts.append(float("nan"))
                continue
            cs, vs = cg[0], vg[0]
            parts.append(vs - cs)
            dw += share * (vs - cs)
        if any(x != x for x in parts):
            print("  %-3s %-38s %9s" % (cid, label, "not run"))
            continue
        rows.append((dw, cid, label, float("nan"), 0.0, 0.0, 0))
        print("  %-3s %-38s %+9.4f %+9.4f %9.5f"
              % (cid, label, parts[0], parts[1], dw))

    if rows:
        rows.sort(reverse=True)
        top = rows[0]
        print(f"\n🔴 STAGE 1 LEADER: {top[1]} ({top[2]}), dW = {top[0]:+.5f}")
        print("  ⛔ Per the pre-registration this candidate -- and ONLY this one --")
        print("     goes to stage 2. If it fails there the search is over and the")
        print("     consensus 60 stands. No second candidate is promoted.")
        if top[0] <= 0:
            print("  ⚠ The leader's dW is <= 0, i.e. the best of 11 variants is not")
            print("    better than the stock list even before a real test. That is")
            print("    §8al's monotone result reproduced by a search, and it is the")
            print("    pre-registered expected outcome.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", action="store_true")
    args = ap.parse_args()
    return cmd_rank() if args.rank else cmd_emit()


if __name__ == "__main__":
    raise SystemExit(main())
