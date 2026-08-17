"""Tag each episode in a policy corpus with the archetype the DEMONSTRATOR played.

The host corpus carries no archetype label, and RUN 2 ("top 20% of Grimmsnarl
games", EVIDENCE §1a follow-up) needs one. This recovers it from the shards
themselves -- no replay re-read -- because `slots` already holds the card ids of
everything in play.

⚠ SIDE MATTERS. `slots` is 12 wide: 0..5 are the demonstrator's own active and
bench, 6..11 are the opponent's (agents/sa/features.py, `slot_ids`). We are
cloning the demonstrator, so an episode is "Grimmsnarl" when GRIMMSNARL IS ON
SLOTS 0..5. Reading all 12 would label every game *against* Grimmsnarl as a
Grimmsnarl game, which in a field where it is the dominant deck is most of them.

Detection is by signature card, not by decklist equality: the Marnie's line has
half a dozen variants in `decks/` (grimmsnarl, _boss, _budew, _xerosic, _g4 ...)
and they are all the same archetype for this purpose.

    python -X utf8 scripts/label_archetypes.py --ds artifacts/pds_hostall \
        --out out/episode_archetypes.csv

Writes `episode_id,archetype`, one row per episode, plus a census to stdout.
"""
from __future__ import annotations

import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Signature card ids per archetype. A card belongs here only if seeing it on a
# player's own side is on its own strong evidence of the archetype -- so the
# evolution line, never the generic trainers (Rare Candy, Boss's Orders and
# Night Stretcher are in half the decks in `decks/`).
ARCHETYPES: dict[str, set[int]] = {
    # decks/grimmsnarl.py: Marnie's Impidimp / Morgrem / Grimmsnarl ex.
    # Impidimp is the basic of the line, so it reaches the bench in essentially
    # every game the deck actually plays -- which is what makes slot recall high.
    "grimmsnarl": {646, 647, 648},
}

# The demonstrator's own slots. See the module docstring -- this is the whole
# correctness argument of the script.
MY_SLOTS = slice(0, 6)


def label_corpus(shards: list[str], archetypes: dict[str, set[int]],
                 use_bags: bool = False, verbose: bool = True
                 ) -> tuple[dict[int, set[str]], int]:
    """episode id -> set of archetypes seen on the demonstrator's side."""
    hits: dict[int, set[str]] = {}
    seen: set[int] = set()
    sig = {name: np.fromiter(ids, dtype=np.int32, count=len(ids))
           for name, ids in archetypes.items()}
    t0 = time.time()
    for i, p in enumerate(shards):
        with np.load(p) as z:
            gid = z["gid"]
            seen.update(gid.tolist())
            cards = [z["slots"][:, MY_SLOTS]]
            if use_bags:
                # my_hand / my_discard are the demonstrator's too, and they
                # catch a line that was drawn or discarded without ever being
                # benched. Ragged, so they are matched per-row via the offsets.
                for nm in ("my_hand", "my_discard"):
                    flat, off = z[f"bag_{nm}_flat"], z[f"bag_{nm}_off"]
                    for name, ids in sig.items():
                        m = np.isin(flat, ids)
                        if not m.any():
                            continue
                        rows = np.searchsorted(off, np.flatnonzero(m),
                                               side="right") - 1
                        for g in np.unique(gid[rows]).tolist():
                            hits.setdefault(g, set()).add(name)
            block = np.concatenate(cards, axis=1)
            for name, ids in sig.items():
                m = np.isin(block, ids).any(axis=1)
                if m.any():
                    for g in np.unique(gid[m]).tolist():
                        hits.setdefault(g, set()).add(name)
        if verbose and (i + 1) % 100 == 0:
            print(f"  scanned {i + 1}/{len(shards)} shards "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return hits, len(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="corpus dir to label")
    ap.add_argument("--out", default="", help="CSV to write (episode_id,archetype)")
    ap.add_argument("--bags", action="store_true",
                    help="also match my_hand/my_discard, not just the board. "
                         "Higher recall, much slower.")
    args = ap.parse_args()

    src = Path(args.ds)
    if not src.is_absolute():
        src = ROOT / src
    shards = sorted(glob.glob(f"{src}/**/shard_*.npz", recursive=True))
    if not shards:
        raise SystemExit(f"no shards under {src}")
    print(f"{len(shards)} shards under {src}", flush=True)

    hits, n_seen = label_corpus(shards, ARCHETYPES, use_bags=args.bags)

    # rule 9: report the census, in episodes, against the total -- a labeller
    # that matches almost nothing must not be mistaken for a rare archetype.
    print(f"\n{n_seen:,} episodes scanned")
    for name in ARCHETYPES:
        k = sum(1 for v in hits.values() if name in v)
        print(f"  {name:14s} {k:7,} ({k / max(n_seen, 1):6.1%})")
    multi = sum(1 for v in hits.values() if len(v) > 1)
    if multi:
        print(f"  ⚠ {multi:,} episodes matched more than one archetype")
    unlabelled = n_seen - len(hits)
    print(f"  {'other':14s} {unlabelled:7,} ({unlabelled / max(n_seen, 1):6.1%})")

    if args.out:
        op = Path(args.out)
        if not op.is_absolute():
            op = ROOT / op
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["episode_id", "archetype"])
            for g in sorted(hits):
                w.writerow([g, "|".join(sorted(hits[g]))])
        print(f"\nwrote {len(hits):,} labelled episodes -> {op}")


if __name__ == "__main__":
    main()
