"""What does `features.py` DROP from the state dict, and does it vary? (day 12)

§8f's method was "read the feature code against a premise nobody had checked".
This does the same thing by ENUMERATION instead of by guessing: it walks real
observations and reports, for every field of `obs['current']` / `obs['select']`
that `featurize()` never reads, how often it varies at a decision point.

Rule 14 (size before you build) applies to features exactly as it applies to
rules: an absent input that is CONSTANT where the decisions happen cannot
explain a single miss, however obviously relevant it sounds. The output is a
per-context distribution for each candidate, so a candidate can be killed for
the price of one scan.

⚠ The day-10/11 candidate list in HANDOFF/ROADMAP ("opponent hand size,
prizes remaining, turn number") is wrong -- all three ARE encoded, at
`features.py` lines 88-99. This script exists so the next list is derived, not
remembered.

    python -X utf8 scripts/p18_missing_state_audit.py --games 400
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents"):
    sys.path.insert(0, str(ROOT / sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()

from cg.api import SelectContext  # noqa: E402

CTX_NAME = {int(getattr(SelectContext, n)): n
            for n in dir(SelectContext) if n.isupper()}

# Everything `featurize()` reads today, so the "dropped" list below is a
# difference and not a memory. Keep in sync with agents/sa/features.py.
READ_STATE = {"turn", "firstPlayer", "players", "supporterPlayed",
              "energyAttached", "yourIndex", "result"}
READ_PLAYER = {"prize", "deckCount", "handCount", "hand", "discard",
               "active", "bench", "poisoned", "burned", "asleep",
               "paralyzed", "confused"}
READ_POKEMON = {"id", "hp", "maxHp", "energies", "appearThisTurn", "tools"}
READ_SELECT = {"type", "minCount", "maxCount", "context", "option"}


def candidates(obs: dict) -> dict[str, object]:
    """The absent inputs, as scalars we can bucket."""
    st = obs["current"]
    sel = obs["select"]
    me = st["yourIndex"]
    mypl, oppl = st["players"][me], st["players"][1 - me]
    stad = st.get("stadium") or []

    def tool_ids(pl):
        out = []
        for pk in [pl["active"][0] if pl["active"] else None] + list(pl["bench"]):
            if pk:
                out += [t["id"] for t in (pk.get("tools") or [])]
        return out

    def evo_depth(pl):
        d = 0
        for pk in [pl["active"][0] if pl["active"] else None] + list(pl["bench"]):
            if pk:
                d = max(d, len(pk.get("preEvolution") or []))
        return d

    return {
        "retreated": bool(st.get("retreated")),
        "stadiumPlayed": bool(st.get("stadiumPlayed")),
        "stadium_id": stad[0]["id"] if stad else 0,
        "turnActionCount": min(int(st.get("turnActionCount") or 0), 20),
        "remainDamageCounter": int(sel.get("remainDamageCounter") or 0),
        "remainEnergyCost": int(sel.get("remainEnergyCost") or 0),
        "sel_effect_card": (sel.get("effect") or {}).get("id", 0),
        "sel_contextCard": (sel.get("contextCard") or {}).get("id", 0)
        if isinstance(sel.get("contextCard"), dict) else 0,
        "my_tools_n": len(tool_ids(mypl)),
        "opp_tools_n": len(tool_ids(oppl)),
        "my_evo_depth": evo_depth(mypl),
        "opp_prize_left": len(oppl["prize"]),   # control: THIS one is encoded
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*",
                    default=["replays/2026-07-29", "replays/2026-07-30"])
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--min-rows", type=int, default=500)
    args = ap.parse_args()

    paths: list[Path] = []
    for d in args.dirs:
        paths += sorted(p for p in (ROOT / d).rglob("*.json")
                        if p.stem.isdigit())
    paths = paths[:args.games]
    if not paths:
        raise SystemExit("no replays found")

    overall: dict[str, Counter] = defaultdict(Counter)
    by_ctx: dict[int, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter))
    ctx_rows: Counter[int] = Counter()
    unread_state: Counter[str] = Counter()
    unread_player: Counter[str] = Counter()
    unread_pk: Counter[str] = Counter()
    unread_sel: Counter[str] = Counter()
    n_rows = 0

    for path in paths:
        try:
            rep = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for v in rep["steps"][0][0].get("visualize") or []:
            obs = v.get("obs")
            if not obs or not obs.get("current") or not obs.get("select"):
                continue
            st, sel = obs["current"], obs["select"]
            if st["result"] != -1 or len(sel.get("option") or []) < 2:
                continue
            if not isinstance(v.get("selected") or v.get("action"), list):
                continue
            n_rows += 1
            unread_state.update(k for k in st if k not in READ_STATE)
            unread_sel.update(k for k in sel if k not in READ_SELECT)
            pl = st["players"][st["yourIndex"]]
            unread_player.update(k for k in pl if k not in READ_PLAYER)
            act = pl["active"][0] if pl["active"] else None
            if act:
                unread_pk.update(k for k in act if k not in READ_POKEMON)
            c = int(sel.get("context") or 0)
            ctx_rows[c] += 1
            vals = candidates(obs)
            for k, val in vals.items():
                overall[k][val] += 1
                by_ctx[c][k][val] += 1

    print(f"\n{n_rows} decision points over {len(paths)} games\n")
    print("=== fields present in the observation that featurize() never reads")
    for name, cnt in (("current", unread_state), ("player", unread_player),
                      ("pokemon", unread_pk), ("select", unread_sel)):
        print(f"  {name:<9}: " + ", ".join(sorted(cnt)))

    print("\n=== how much does each dropped field VARY at a decision point?")
    print(f"{'candidate':<22}{'distinct':>9}{'modal share':>13}   top values")
    for k, cnt in overall.items():
        modal = max(cnt.values()) / n_rows
        top = ", ".join(f"{v}:{n/n_rows:.0%}" for v, n in cnt.most_common(4))
        print(f"{k:<22}{len(cnt):>9}{modal:>13.1%}   {top}")

    print("\n=== per-context modal share (a candidate is dead where this is ~100%)")
    ks = [k for k in overall if len(overall[k]) > 1]
    hdr = "".join(f"{k[:11]:>13}" for k in ks)
    print(f"{'context':<28}{'rows':>7}{hdr}")
    for c, n in ctx_rows.most_common():
        if n < args.min_rows:
            continue
        cells = "".join(f"{max(by_ctx[c][k].values())/n:>13.0%}" for k in ks)
        print(f"{CTX_NAME.get(c, str(c)):<28}{n:>7}{cells}")
    print("\nread a column as: how often the field takes its most common value "
          "in that context. 100% = constant = cannot explain any miss there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
