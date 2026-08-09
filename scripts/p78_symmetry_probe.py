"""Is the shipped net's decision STABLE under semantically-null relabelings?

**Why this exists.** Every feature audit this project has run (`p18`, §8y/§8z,
§8ab) asked "what is in the observation that `featurize` does not read?" None
asked the dual question: **"what does the net read that carries no game
meaning?"** A bench SLOT NUMBER is the clearest instance. Moving a Pokemon from
bench slot 1 to bench slot 3 changes nothing about the game, but it changes
`opt["index"]` — which §8f encoded deliberately, and was worth +115 Elo, because
it was the only thing distinguishing two options naming two copies of one card.

⇒ If the net's choice **changes identity** when the bench is merely relabelled,
it is deciding partly on slot noise. That is both a defect and an opportunity:
averaging the logits over K relabelings is a free variance reduction that costs
~1 ms x K out of a 600 s budget we spend 0.1 s of.

**Two arms, and the first is a POSITIVE CONTROL, not a result:**
  * `optorder` — permute the ORDER of the option list only. The net pools
    options with mean/max deep-sets (§8aa) and per-option features do not read
    list position, so this **must** measure ~0. A non-zero reading here means
    the remapping harness below is broken and the bench arm is meaningless.
  * `bench` — permute our own bench array AND rewrite every option index that
    points into it. Semantically the identical position.

    python -X utf8 scripts/p78_symmetry_probe.py --games 40 --k 8
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from p72_loss_autopsy import _nm, _records  # noqa: E402
from sa import policynet as pnet  # noqa: E402

BENCH = 5


def _opt_key(obs: dict, j: int) -> tuple:
    """A permutation-INVARIANT identity for option j: what it actually names.

    Comparing raw indices across a relabelling is meaningless, so the argmax is
    compared on (type, area, owner, card id, hp, energy count) instead.
    """
    from sa.optfeat import option_features
    sel, st = obs["select"], obs["current"]
    o = (sel.get("option") or [])[j]
    cid = 0
    try:
        cid = int(option_features(obs, o)[1] or 0)
    except Exception:  # noqa: BLE001
        pass
    hp = ecount = -1
    area, idx = o.get("area"), o.get("index") or 0
    pi = o.get("playerIndex")
    pi = st.get("yourIndex") if pi is None else pi
    try:
        pl = st["players"][pi]
        arr = pl["active"] if area == 4 else (pl["bench"] if area == BENCH else None)
        if arr is not None and 0 <= idx < len(arr) and arr[idx]:
            hp, ecount = arr[idx]["hp"], len(arr[idx]["energies"])
    except (KeyError, IndexError, TypeError):
        pass
    return (o.get("type"), area, pi, cid, hp, ecount)


def _permute_bench(obs: dict, seat: int, perm: list[int]) -> dict | None:
    """Relabel seat's bench by `perm`, rewriting every option that indexes it.

    `perm[new] = old`. Returns None when the select references a bench slot the
    permutation cannot express (nothing here is worth a silent wrong answer).
    """
    out = copy.deepcopy(obs)
    st = out["current"]
    bench = st["players"][seat]["bench"]
    n = len(bench)
    if n < 2 or sorted(perm) != list(range(n)):
        return None
    st["players"][seat]["bench"] = [bench[perm[i]] for i in range(n)]
    inv = [0] * n
    for new, old in enumerate(perm):
        inv[old] = new
    for o in (out["select"].get("option") or []):
        pi = o.get("playerIndex")
        pi = st.get("yourIndex") if pi is None else pi
        if o.get("area") == BENCH and pi == seat:
            k = o.get("index") or 0
            if not 0 <= k < n:
                return None
            o["index"] = inv[k]
    return out


def _permute_optorder(obs: dict, perm: list[int]) -> dict:
    out = copy.deepcopy(obs)
    opts = out["select"].get("option") or []
    out["select"]["option"] = [opts[i] for i in perm]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="replays/submission_v5_s2")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--k", type=int, default=8, help="relabelings per decision")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    net = pnet.get()
    if net is None:
        print("🔴 no policy net loaded")
        return 1
    rng = random.Random(args.seed)

    stats: Counter = Counter()
    flip_by_ctx: Counter = Counter()
    seen_by_ctx: Counter = Counter()
    margins: list[float] = []
    flip_margins: list[float] = []

    paths = sorted((ROOT / args.dir).glob("*.json"))[:args.games]
    for path in paths:
        try:
            rep = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for rec in _records(rep):
            obs, sel = rec["obs"], rec["sel"]
            st = obs["current"]
            opts = sel.get("option") or []
            if len(opts) < 2 or st.get("result", -1) != -1:
                continue
            seat = st.get("yourIndex")
            if seat not in (0, 1):
                continue
            try:
                base = net.scores(obs)
            except Exception:  # noqa: BLE001
                continue
            if base is None or len(base) != len(opts):
                continue
            b_arg = int(np.argmax(base))
            b_key = _opt_key(obs, b_arg)
            srt = np.sort(base)[::-1]
            margin = float(srt[0] - srt[1])
            margins.append(margin)
            ctx = sel.get("context")

            # --- arm 1: option ORDER (positive control, must read ~0)
            for _ in range(args.k):
                perm = list(range(len(opts)))
                rng.shuffle(perm)
                o2 = _permute_optorder(obs, perm)
                try:
                    s2 = net.scores(o2)
                except Exception:  # noqa: BLE001
                    continue
                stats["optorder trials"] += 1
                if _opt_key(o2, int(np.argmax(s2))) != b_key:
                    stats["optorder FLIPS"] += 1

            # --- arm 2: our own BENCH relabelling
            nb = len(st["players"][seat]["bench"])
            if nb < 2:
                continue
            seen_by_ctx[ctx] += 1
            flipped_here = False
            for _ in range(args.k):
                perm = list(range(nb))
                rng.shuffle(perm)
                o2 = _permute_bench(obs, seat, perm)
                if o2 is None:
                    continue
                try:
                    s2 = net.scores(o2)
                except Exception:  # noqa: BLE001
                    continue
                stats["bench trials"] += 1
                if _opt_key(o2, int(np.argmax(s2))) != b_key:
                    stats["bench FLIPS"] += 1
                    flipped_here = True
            if flipped_here:
                flip_by_ctx[ctx] += 1
                flip_margins.append(margin)

    print(f"\n=== SYMMETRY PROBE — {len(paths)} games, k={args.k} relabelings ===")
    ot, of = stats["optorder trials"], stats["optorder FLIPS"]
    bt, bf = stats["bench trials"], stats["bench FLIPS"]
    print(f"\nARM 1 (POSITIVE CONTROL) option-list order")
    print(f"  {of}/{ot} = {of/max(ot,1):.3%} of relabelings change the choice")
    print("  ✅ near 0 means the harness is sound" if of / max(ot, 1) < 0.005
          else "  🔴 NON-ZERO — the remapping harness is broken; arm 2 is void")

    print(f"\nARM 2 our own BENCH slot relabelling (semantically null)")
    print(f"  {bf}/{bt} = {bf/max(bt,1):.3%} of relabelings change the choice")
    dec = sum(seen_by_ctx.values())
    nflip = sum(flip_by_ctx.values())
    print(f"  {nflip}/{dec} = {nflip/max(dec,1):.1%} of DECISIONS are unstable"
          f" under at least one of {args.k} relabelings")
    if margins:
        print(f"\n  logit margin (top1 - top2): all decisions"
              f" median {np.median(margins):.3f}")
    if flip_margins:
        print(f"                              unstable ones"
              f" median {np.median(flip_margins):.3f}")
        print("  ⇒ if the unstable ones are the near-ties, the flips are cheap;")
        print("    if they are not, the net is deciding on slot noise.")
    if flip_by_ctx:
        print("\n  most unstable contexts (flipped decisions / decisions seen):")
        for c, v in flip_by_ctx.most_common(8):
            print(f"    ctx {c:<3}{v:>6} / {seen_by_ctx[c]:<6}"
                  f" = {v/max(seen_by_ctx[c],1):>6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
