"""Would swapping a card break the clone? Audit the exposure BEFORE editing the deck.

**The user's question, day 15:** *"Would changing a card dramatically decrease
our strength because the agent wouldn't know how to play that new card?"*

It is the right worry and it has a mechanism, so it can be measured instead of
guessed. The net encodes a card **twice**:

  1. **Derived properties** -- HP, max HP, damage fraction, stage, ex/megaEx/
     tera, retreat cost, prize value, attached energy, cost satisfaction,
     `best_estimated_damage` (`features._slot_feats`, `optfeat.option_features`
     indices 25..36). **These are computed from the card database for ANY card,
     including one that appeared in zero training games.** A brand-new card gets
     correct values here on its first appearance.
  2. **Card-id embeddings** -- `slot_emb`, `bag_emb`, `card_emb` (1300 x 16) and
     `atk_emb` (1600 x 16). A row here is only meaningful if that id appeared in
     training; **a never-seen id keeps its random initialisation and injects
     noise** into a summed bag.

⇒ **Channel 1 degrades gracefully and channel 2 does not.** So the real question
is not "is the card new to the deck" but **"is the card new to the CORPUS"** --
and our corpus is 2,810 games of the whole field, both seats, so it contains an
enormous number of cards we have never played ourselves.

This script answers, for any candidate card id:
  * did it appear in training at all, and how often;
  * is its embedding row still at initialisation (a direct check on the net,
    independent of the corpus);
  * how many of our own decisions would even involve it (rule 14: size it).

    python -X utf8 scripts/p22_deck_change_risk.py --net out/policy_v5.npz
    python -X utf8 scripts/p22_deck_change_risk.py --cards 1244 1259 --net out/policy_v5.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()
from sa import cards as cdb  # noqa: E402


# ⚠ NAME THE CARD-ID ARRAYS EXPLICITLY. Pattern-matching the key names picks up
# `bag_my_hand_off` and `opt_off`, which are int64 OFFSETS running to ~400,000 --
# they would be counted as card ids and swamp the table with garbage. `opt_attack`
# is also excluded: attack ids live in a separate 1600-wide space.
ANY_SIDE = ("slots", "xslots", "bag_my_hand_flat", "bag_my_discard_flat",
            "bag_opp_discard_flat")
OUR_OPTION = ("opt_card",)      # cards WE were offered as a choice


def corpus_card_counts(ds: Path) -> tuple[Counter, Counter]:
    """(seen anywhere, seen as one of OUR options).

    The split matters more than the totals. A card can be all over the corpus
    because opponents played it while **our** seat was never once offered it as
    a choice -- and it is the second column that speaks to "would the net know
    how to PLAY it".
    """
    anyc: Counter = Counter()
    ourc: Counter = Counter()
    for shard in sorted(ds.glob("*/shard_*.npz")):
        z = np.load(shard, allow_pickle=True)
        for keys, dest in ((ANY_SIDE, anyc), (OUR_OPTION, ourc)):
            for key in keys:
                if key not in z.files:
                    continue
                a = z[key]
                if a.dtype.kind not in "iu":
                    continue
                vals, cnts = np.unique(a.ravel(), return_counts=True)
                for v, n in zip(vals.tolist(), cnts.tolist()):
                    if v and 0 < int(v) < 1300:
                        dest[int(v)] += int(n)
    for cid, n in ourc.items():       # our options are also "seen anywhere"
        anyc[cid] += n
    return anyc, ourc


def emb_separation(emb: np.ndarray, seen: set[int]) -> str:
    """Report whether trained and untrained rows are even distinguishable.

    🔴 **The first version of this file tried to DETECT untrained rows from the
    net alone**, on the theory that a row which never received gradient sits at
    its init, so untrained rows would share one tight norm. **That is wrong:
    the init is i.i.d. random, so every untrained row has its OWN random norm
    and there is no cluster to find.** Measured on `policy_v5.npz`, seen rows
    average 4.008 and unseen 3.952, with 1,032 of 1,166 unseen rows sitting
    inside the seen rows' 5–95 percentile band — the heuristic separated
    nothing, and the column it printed ("row trained: yes") was meaningless for
    every card.

    **Same failure as `p20_recorder_equivalence`'s first version, one day
    apart: a check that could not have come out the other way.** The corpus id
    set is ground truth and needs no inference — this function now only reports
    the (non-)separation, so the mistake stays visible instead of being quietly
    deleted.
    """
    n = np.linalg.norm(emb, axis=1)
    m = np.zeros(len(emb), dtype=bool)
    m[np.fromiter(seen, dtype=int)] = True
    lo, hi = np.percentile(n[m], [5, 95])
    inside = int(((n[~m] >= lo) & (n[~m] <= hi)).sum())
    return (f"seen {n[m].mean():.3f} vs unseen {n[~m].mean():.3f}; "
            f"{inside}/{int((~m).sum())} unseen rows inside the seen 5-95 band "
            f"-> NOT separable, use the corpus set")


def name(cid: int) -> str:
    try:
        return str(cdb.card(cid).get("name") or f"#{cid}")
    except Exception:  # noqa: BLE001
        return f"#{cid}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="out/policy_v5.npz")
    ap.add_argument("--ds", default="artifacts/pds_v4")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--cards", type=int, nargs="*", default=[],
                    help="candidate card ids to score (default: audit the "
                         "whole card pool and rank it)")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    print(f"  corpus  {args.ds}")
    counts, ourc = corpus_card_counts(Path(args.ds))
    print(f"  {len(counts):,} distinct card ids appear anywhere "
          f"({sum(counts.values()):,} occurrences)")
    print(f"  {len(ourc):,} distinct card ids were ever OUR OWN option "
          f"({sum(ourc.values()):,} occurrences)")

    net = np.load(args.net)
    print(f"  net     {args.net}")
    seen = set(counts)
    for k in ("slot_emb", "bag_emb", "card_emb"):
        if k in net.files:
            print(f"    {k}: {emb_separation(net[k], seen)}")
    print(f"  🔴 {1300 - len(seen):,} of 1,300 embedding rows in EACH table "
          f"never received a\n     gradient and are still random init. "
          f"A card outside the {len(seen)} below injects\n     three random "
          f"16-d vectors; only the DERIVED features stay correct.")

    at_init = np.ones(1300, dtype=bool)
    at_init[np.fromiter(seen, dtype=int)] = False

    import importlib
    mod = importlib.import_module(f"decks.{args.deck}")
    ours = set(mod.DECKLIST)

    print(f"\n=== OUR OWN 60 ({args.deck}) -- the baseline for 'well known' ===")
    mine = sorted(ours, key=lambda c: ourc.get(c, 0))
    print(f"  {'card':<34}{'id':>6}{'as OUR option':>15}{'anywhere':>11}"
          f"{'row trained':>13}")
    for cid in mine[:5]:
        print(f"  {name(cid):<34}{cid:>6}{ourc.get(cid,0):>15,}"
              f"{counts.get(cid,0):>11,}"
              f"{('no' if at_init[cid] else 'yes'):>13}")
    ours_min = min(ourc.get(c, 0) for c in ours)
    ours_med = int(np.median([ourc.get(c, 0) for c in ours]))
    print(f"  ^ the five we ourselves are offered LEAST often. "
          f"min {ours_min:,}, median {ours_med:,}")
    print("  🔴 THIS IS THE BAR. A card the net handles fine today may sit at "
          "only a\n     few hundred of its own decisions, so 'the net has never "
          "seen it' is a\n     matter of degree, not a cliff.")

    if args.cards:
        cand = list(args.cards)
    else:
        # everything the corpus knows that we do NOT play -- the swap-in pool
        cand = [c for c in counts if c not in ours]
        cand.sort(key=lambda c: -counts[c])
        cand = cand[:args.top]

    print(f"\n=== CANDIDATE SWAP-INS -- how exposed is each one? ===")
    print("  Compared against the WEAKEST card we already play. A candidate at")
    print("  or above that line is no more off-distribution than something the")
    print("  net handles every game.")
    print(f"\n  {'card':<34}{'id':>6}{'as OUR option':>15}{'anywhere':>11}"
          f"{'vs our min':>12}{'trained':>9}{'risk':>9}")
    for cid in cand:
        mine_hits = ourc.get(cid, 0)
        any_hits = counts.get(cid, 0)
        ratio = mine_hits / max(ours_min, 1)
        trained = "no" if (cid < 1300 and at_init[cid]) else "yes"
        if trained == "no" or any_hits == 0:
            risk = "HIGH"
        elif mine_hits == 0:
            risk = "MEDIUM"      # net knows the card, never chose it
        elif ratio >= 1.0:
            risk = "low"
        elif ratio >= 0.1:
            risk = "medium"
        else:
            risk = "HIGH"
        print(f"  {name(cid):<34}{cid:>6}{mine_hits:>15,}{any_hits:>11,}"
              f"{ratio:>11.2f}x{trained:>9}{risk:>9}")

    print("\n=== WHAT THIS DOES AND DOES NOT SETTLE ===")
    print("  Settles: whether the net has ever SEEN the card, which is the")
    print("    mechanism behind 'it won't know how to play it'.")
    print("  Does NOT settle: whether it plays the card WELL. Corpus exposure")
    print("    is necessary, not sufficient -- the card may only ever have been")
    print("    seen in an opponent's hands, where our seat never chose it.")
    print("  ⇒ Any real swap still needs an arena A/B at n>=2000 against the")
    print("    re-weighted anchors (§8ac), with the seed floor carried in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
