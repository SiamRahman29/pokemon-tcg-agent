"""Emit embedding-ablated copies of a policy npz -- no retraining.

The question this answers: does the *identity* channel (which card is this)
carry any decision weight, or are the four embedding tables decoration on top
of the dense mechanical features?

Zeroing a table is the obvious ablation and the wrong one: it moves the input
distribution the downstream layers were trained against, so degradation from
"identity destroyed" cannot be told apart from degradation from "activations
off their training scale". Permutation is the tight control -- the exact same
multiset of row vectors goes in, only the card->row assignment is scrambled.

Modes:

    perm_seen   permute rows WITHIN the seen-id set (p53 census). Holds
                "trainedness" constant; the sharpest identity test.
    perm_all    permute every row. Adds the untrained-row confound but needs
                no census.
    zero        zero every row. The loose ablation, kept for contrast.
    copy        change nothing. The serialisation control -- it round-trips
                through the same dict(np.load) -> np.savez path as the real
                arms, so a dtype or key regression in this script shows up as
                a losing control instead of masquerading as an ablation
                effect. Must score ~0.500 against its own source net.

Row 0 is never touched in any mode: `slot_emb[0]` is the empty/unresolved slot
and the net drove it down on its own (norm 2.337 against a 3.96 table
mean), so it encodes "nothing here", not a card identity. Scrambling it would
confound the result with a board-occupancy signal.

    python -X utf8 scripts/p54_emb_ablate.py --mode perm_seen
    python -X utf8 scripts/p54_emb_ablate.py --mode perm_seen --tables slot_emb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

TABLES = ("slot_emb", "bag_emb", "card_emb", "atk_emb")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ablate(w: np.ndarray, mode: str, seen: list[int],
           rng: np.random.Generator) -> np.ndarray:
    out = w.copy()
    if mode == "copy":
        return out
    if mode == "zero":
        out[1:] = 0.0
        return out
    if mode == "perm_all":
        idx = np.arange(1, w.shape[0])
    elif mode == "perm_seen":
        idx = np.array([i for i in seen if i != 0], dtype=np.int64)
    else:
        raise SystemExit(f"unknown mode {mode}")
    if idx.size < 2:
        return out
    # derangement-ish: reshuffle until no row keeps its own id (cheap for the
    # sizes here; a fixed point would silently weaken the ablation)
    for _ in range(64):
        perm = rng.permutation(idx)
        if not np.any(perm == idx):
            break
    out[idx] = w[perm]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="out/policy_v5.npz")
    ap.add_argument("--vocab", default="out/emb/vocab.json")
    ap.add_argument("--mode", default="perm_seen",
                    choices=["perm_seen", "perm_all", "zero", "copy"])
    ap.add_argument("--tables", default="all",
                    help="comma list of tables, or 'all'")
    # Global permutation scrambles our OWN 19 cards too, which is far harsher
    # than the situation we actually face -- our deck is 19/19 in vocabulary
    # and every lookup on it lands on a heavily-trained row. Excluding a deck
    # holds our own-card identity fixed and scrambles only cards we can see
    # but do not own, which is exactly "we cannot identify opponent cards".
    ap.add_argument("--exclude-deck", default=None,
                    help="decks/<name>.py whose card ids are left untouched")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output npz path")
    args = ap.parse_args()

    src = ROOT / args.net
    if not src.exists():
        raise SystemExit(f"net not found: {src}")
    z = dict(np.load(src))
    # A v7 net's rows are vocabulary indices, not card ids, so the p53 census
    # this script permutes WITHIN would scramble the wrong rows -- and with a
    # ~137-row table it would also touch UNK and PAD. Refuse rather than emit a
    # net whose ablation means something other than what the arm claims.
    if "vocab_slot_emb" in z:
        raise SystemExit(f"{src.name} is a --vocab (v7) net: its rows are "
                         "remapped indices, not card ids. p54's census-based "
                         "permutation does not apply; regenerate the census "
                         "against the remapped id space first.")

    which = TABLES if args.tables == "all" else tuple(
        t.strip() for t in args.tables.split(","))
    for t in which:
        if t not in z:
            raise SystemExit(f"{src.name} has no tensor {t}")

    vocab = {}
    if args.mode == "perm_seen":
        vp = ROOT / args.vocab
        if not vp.exists():
            raise SystemExit(f"{vp} missing -- run scripts/p53_emb_vocab.py")
        vocab = json.loads(vp.read_text(encoding="utf-8"))["tables"]

    keep: set[int] = set()
    if args.exclude_deck:
        import importlib
        mod = importlib.import_module(f"decks.{args.exclude_deck}")
        keep = set(int(c) for c in mod.DECK.decklist)
        print(f"holding {len(keep)} card ids from decks/{args.exclude_deck}.py "
              f"fixed")

    rng = np.random.default_rng(args.seed)
    print(f"source {args.net}  sha256={sha(src)[:16]}")
    for t in which:
        w = z[t]
        seen = sorted(int(k) for k in vocab.get(t, {})) if vocab else []
        if keep:
            seen = [i for i in seen if i not in keep]
        new = ablate(w, args.mode, seen, rng)
        moved = int(np.any(new != w, axis=1).sum())
        z[t] = new
        print(f"  {t:9s} rows={w.shape[0]:5d} emb={w.shape[1]:3d} "
              f"perm_pool={len(seen):4d} changed={moved:5d}")

    tag = args.tables if args.tables != "all" else "all"
    if args.exclude_deck:
        tag += f"__keep-{args.exclude_deck}"
    name = args.out or f"out/emb/{src.stem}__{args.mode}__{tag}__s{args.seed}.npz"
    dst = ROOT / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **z)

    # The dim guard silently returns None for a net this code cannot feed, and
    # the agent then plays random-legal while looking healthy. Fail here.
    from ptcg.env import sdk
    sdk.load()
    from sa import policynet as pn
    if pn.load(dst) is None:
        raise SystemExit(f"{dst} FAILS the dim guard")
    print(f"wrote {dst}  sha256={sha(dst)[:16]}  (dim guard OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
