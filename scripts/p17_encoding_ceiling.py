"""How much of the clone's ~30% residual is PROVABLY the encoding? (day 12)

§8w ruled out capacity (8.2x params, no gain) and §8u ruled out demonstrator
selection, leaving "the residual is the encoding" as a conclusion BY
ELIMINATION. This measures it directly, with no net involved at all.

Two ceilings, both computed from the shards' own inputs:

  1. WITHIN-ROW COLLISIONS -- the §8f defect, counted.
     Two options whose (opt_dense, card_id, attack_id, target_id) are
     bitwise identical get identical logits from ANY net that reads only
     those inputs. If the demonstrator's chosen option sits in a tie group
     of size g, no such net can beat 1/g on that row. Summing 1/g over rows
     is a hard upper bound on top-1 agreement for THIS feature layout.
     This is exactly the defect `opt["index"]` fixed in §8f -- now measured
     over the whole corpus instead of found by reading code.

  2. CROSS-ROW LABEL DISAGREEMENT -- the Bayes floor.
     Rows whose FULL net input is bitwise identical (state dense + slot ids
     + bag contents + select features + the multiset of option encodings)
     are indistinguishable to any net. Where two such rows carry different
     chosen options, the demonstrators themselves disagreed. The modal
     label's share is an upper bound on achievable agreement, and 1 - that
     is irreducible policy entropy -- a budget no feature can recover
     UNLESS the feature breaks the tie (i.e. the states differ in some way
     the encoding drops).

The two bound different things and both matter: (1) says "the answer is not
expressible", (2) says "there is no single right answer here".

    python -X utf8 scripts/p17_encoding_ceiling.py --ds artifacts/pds_v3r
    python -X utf8 scripts/p17_encoding_ceiling.py --ds artifacts/pds_v3r --opt-cols 25

`--opt-cols 25` re-runs ceiling (1) against the v2 option layout, which is
the §8f control: the ceiling it reports is what the pre-B1 net was fighting.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from hashlib import blake2b
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents"):
    sys.path.insert(0, str(ROOT / sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()

from cg.api import SelectContext  # noqa: E402

CTX_NAME = {int(getattr(SelectContext, n)): n
            for n in dir(SelectContext) if n.isupper()}
BAGS = ("my_hand", "my_discard", "opp_discard")
N_OPTION_TYPES = 17


def h(*parts: bytes) -> bytes:
    d = blake2b(digest_size=12)
    for p in parts:
        d.update(p)
    return d.digest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="artifacts/pds_v3r")
    ap.add_argument("--opt-cols", type=int, default=0,
                    help="truncate per-option dense to this width (25 = the "
                         "v2 layout, i.e. the §8f control)")
    ap.add_argument("--min-rows", type=int, default=200)
    ap.add_argument("--examples", type=int, default=0,
                    help="print this many example tie groups per context")
    args = ap.parse_args()

    paths = sorted((ROOT / args.ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")

    # ---- pass 1: within-row option collisions --------------------------
    rows_c: Counter[int] = Counter()          # rows per context
    tied_c: Counter[int] = Counter()          # rows whose CHOSEN option is tied
    ceil_c: defaultdict[int, float] = defaultdict(float)
    opts_c: Counter[int] = Counter()          # total options seen
    tiedtype: Counter[int] = Counter()        # option type of tied chosen
    examples: defaultdict[int, list] = defaultdict(list)

    # keyed by full-input hash -> Counter of chosen-option hashes
    groups: defaultdict[bytes, Counter[bytes]] = defaultdict(Counter)
    grp_ctx: dict[bytes, int] = {}

    for path in paths:
        z = np.load(path)
        off = z["opt_off"]
        n = len(z["gid"])
        od = z["opt_dense"]
        if args.opt_cols:
            od = od[:, :args.opt_cols]
        card, atk, tgt = z["opt_card"], z["opt_attack"], z["opt_target"]
        chosen = z["opt_chosen"]
        ctx_all = np.rint(z["seld"][:, 13] * 50.0).astype(int)
        dense, slots, seld = z["dense"], z["slots"], z["seld"]
        bag_flat = {nm: z[f"bag_{nm}_flat"] for nm in BAGS}
        bag_off = {nm: z[f"bag_{nm}_off"] for nm in BAGS}

        # one byte-key per option, vectorised: view the concatenated bytes
        raw = np.concatenate(
            [np.ascontiguousarray(od).view(np.uint8).reshape(len(od), -1),
             np.ascontiguousarray(card).view(np.uint8).reshape(len(od), -1),
             np.ascontiguousarray(atk).view(np.uint8).reshape(len(od), -1),
             np.ascontiguousarray(tgt).view(np.uint8).reshape(len(od), -1)],
            axis=1)
        raw = np.ascontiguousarray(raw)
        keys = raw.view([("k", np.void, raw.shape[1])]).reshape(-1)
        _, ids = np.unique(keys, return_inverse=True)   # local integer ids

        for r in range(n):
            a, b = off[r], off[r + 1]
            ch = chosen[a:b]
            if ch.sum() != 1:        # top-1 is only defined for single-choice
                continue
            c = int(ctx_all[r])
            k_ids = ids[a:b]
            ci = int(np.argmax(ch))
            g = int((k_ids == k_ids[ci]).sum())
            rows_c[c] += 1
            opts_c[c] += b - a
            ceil_c[c] += 1.0 / g
            if g > 1:
                tied_c[c] += 1
                t = int(np.argmax(od[a + ci, :N_OPTION_TYPES])) \
                    if od[a + ci, :N_OPTION_TYPES].max() > 0 else -1
                tiedtype[t] += 1
                if args.examples and len(examples[c]) < args.examples:
                    examples[c].append((str(path.parent.name), r, g, b - a, t))

            # ---- pass 2 bookkeeping: full-input identity ----------------
            bag_bytes = []
            for nm in BAGS:
                o = bag_off[nm]
                bag_bytes.append(np.sort(bag_flat[nm][o[r]:o[r + 1]]).tobytes())
            opt_set = b"".join(sorted(keys[a + i].tobytes()
                                      for i in range(b - a)))
            gk = h(dense[r].tobytes(), slots[r].tobytes(), seld[r].tobytes(),
                   *bag_bytes, opt_set)
            groups[gk][keys[a + ci].tobytes()] += 1
            grp_ctx[gk] = c

    # ---- report 1 -------------------------------------------------------
    label = f"{args.ds}" + (f"  [opt-cols={args.opt_cols}]"
                            if args.opt_cols else "  [full v3 layout]")
    print(f"\n=== 1. WITHIN-ROW OPTION COLLISIONS -- {label}\n")
    print(f"{'context':<28}{'rows':>8}{'opts/row':>10}{'tied rows':>11}"
          f"{'ceiling':>10}")
    tot_rows = tot_ceil = tot_tied = 0.0
    for c in sorted(rows_c, key=lambda c: -(rows_c[c] - ceil_c[c])):
        nr = rows_c[c]
        tot_rows += nr
        tot_ceil += ceil_c[c]
        tot_tied += tied_c[c]
        if nr < args.min_rows:
            continue
        print(f"{CTX_NAME.get(c, str(c)):<28}{nr:>8}{opts_c[c]/nr:>10.1f}"
              f"{tied_c[c]/nr:>10.1%} {ceil_c[c]/nr:>9.1%}")
    print(f"\n{int(tot_rows)} single-choice rows; "
          f"chosen option is bitwise-tied on {tot_tied/tot_rows:.1%} of them; "
          f"TOP-1 CEILING = {tot_ceil/tot_rows:.1%}")
    if tiedtype:
        print("option type of the tied chosen option: "
              + ", ".join(f"{t}:{n}" for t, n in tiedtype.most_common(8)))
    for c, ex in examples.items():
        for e in ex:
            print(f"  eg {CTX_NAME.get(c, c)}: shard={e[0]} row={e[1]} "
                  f"tie={e[2]} of {e[3]} options, type={e[4]}")

    # ---- report 2 -------------------------------------------------------
    dup = {k: v for k, v in groups.items() if sum(v.values()) > 1}
    n_dup_rows = sum(sum(v.values()) for v in dup.values())
    print(f"\n=== 2. CROSS-ROW LABEL DISAGREEMENT (the Bayes floor)\n")
    print(f"{len(groups)} distinct full-input states over {int(tot_rows)} rows; "
          f"{len(dup)} appear more than once, covering {n_dup_rows} rows "
          f"({n_dup_rows/tot_rows:.1%})")
    if not dup:
        print("no repeated inputs -- the floor is unmeasurable this way")
        return 0
    modal = sum(max(v.values()) for v in dup.values())
    print(f"on those rows the demonstrators agree with the modal choice "
          f"{modal/n_dup_rows:.1%} of the time "
          f"=> irreducible label noise >= {1 - modal/n_dup_rows:.1%}")
    by_ctx: defaultdict[int, list] = defaultdict(lambda: [0, 0])
    for k, v in dup.items():
        s = sum(v.values())
        by_ctx[grp_ctx[k]][0] += s
        by_ctx[grp_ctx[k]][1] += max(v.values())
    print(f"\n{'context':<28}{'dup rows':>10}{'modal share':>13}")
    for c, (s, m) in sorted(by_ctx.items(), key=lambda kv: -kv[1][0]):
        if s < 50:
            continue
        print(f"{CTX_NAME.get(c, str(c)):<28}{s:>10}{m/s:>13.1%}")
    print("\n⚠ ceiling (1) bounds ANY net on this layout; floor (2) is measured "
          "only on inputs that repeat exactly, which are the cheap/early ones -- "
          "read it as a lower bound on noise in that subpopulation, not the whole.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
