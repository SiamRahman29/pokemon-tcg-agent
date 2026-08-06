"""Which slots are MATCHUP-SPECIFIC? The instrument §8al asked for and never got.

**Why this exists (day 18).** §8al retired the guess-a-swap deck method and named
its successor: *"the next deck programme needs a MATCHUP-STRATIFIED SEARCH DESIGN
over the whole slot ranking"*, because **all four deck A/Bs so far were
mirror-only**, which flatters any variant that cuts mirror-dead tech (Tool
Scrapper: **0.00 plays per mirror game**, but drawn in 81% of real games) and
cannot judge a card aimed anywhere else.

`p33_anchor_resolution.py` then priced the stratified design and found the case
for it is **entirely about bias, not precision**: the same games spent
mirror-only measure `Delta_mirror` more tightly than a seven-cell split measures
the field-weighted `W`. Stratifying is only worth its cost **for cards whose
liveness differs by matchup** -- and for cards whose liveness does not, a mirror
A/B is both cheaper and unbiased.

So the design reduces to one question this repo cannot currently answer:
**per card, per matchup, how often is the slot live?** `p25_deck_slot_audit.py`
answers it pooled over real ladder games, which is exactly the pooling that hides
the effect. This script un-pools it.

**How.** Our agent is wrapped so every `select` is tallied as it happens -- no
replay files (§8ad's recorder writes multi-MB per game, which is 7 GB at the n
needed here), no post-hoc parsing, no seat arithmetic to get wrong (rule 18: the
wrapper travels with the agent, so it tallies OUR selects whichever seat we are
in). Option -> card resolution uses `optfeat.option_features`, the same resolver
the net itself uses and the one `p25` uses, so the two are comparable.

    python -X utf8 scripts/p34_matchup_liveness.py --games 300
    python -X utf8 scripts/p34_matchup_liveness.py --games 40 --anchors mirror crustle

⚠ This measures liveness, NOT value. A card played 0.1 times a game can still win
those games (Boss's Orders -- the §8e trap, restated in `p25`'s own footer).
Liveness caps what a swap can pay and says WHERE it could pay; it never says a
card is bad.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import harness, sdk  # noqa: E402
sdk.load()
from arena import build_agent, resolve_deck  # noqa: E402
from sa import cards as cdb, optfeat  # noqa: E402
from decks import grimmsnarl  # noqa: E402

NET = "out/policy_v5.npz"

# The current anchor set (EVIDENCE §8ap), each as (label, our spec, their spec,
# their deck).  `mirror` is our own net on our own 60 -- the only cell where a
# deck A/B is a direct head-to-head.
ANCHORS = {
    "mirror":     (f"bc:v5,net={NET}", f"bc:v5,net={NET}", "grimmsnarl"),
    "v10":        (f"bc:v5,net={NET}", "rule:v10,noS",     "lucario_v10"),
    "archaludon": (f"bc:v5,net={NET}", "rule:archaludon",  "archaludon_ex"),
    "alakazam5":  (f"bc:v5,net={NET}", "rule:alakazam5",   "alakazam5"),
    "dragapult":  (f"bc:v5,net={NET}", "rule:dragapult",   "dragapult_ex"),
    "garchomp":   (f"bc:v5,net={NET}", f"bc:v5,net={NET}", "cynthia_garchomp"),
    "crustle":    (f"bc:v5,net={NET}", "rule:crustle",     "crustle_v1"),
}
# EVIDENCE §8ac, the band-aware census of our own ladder replays.
SHARES = {"mirror": 0.333, "v10": 0.040, "archaludon": 0.080, "alakazam5": 0.220,
          "dragapult": 0.053, "garchomp": 0.067, "crustle": 0.067}


def name_of(cid: int) -> str:
    return (cdb.card(cid) or {}).get("name", str(cid))


class Tally:
    """Per-card counters for one matchup, filled in by the agent wrapper."""

    def __init__(self, deck: dict[int, int]) -> None:
        self.deck = deck
        self.offered = Counter()   # selects where the card was one of our options
        self.taken = Counter()     # selects where we chose it
        self.drawn_games = Counter()  # games where it reached our hand at all
        self.games = 0
        self.selects = 0
        self._seen_hand: set[int] = set()

    def end_game(self) -> None:
        self.games += 1
        for cid in self._seen_hand:
            self.drawn_games[cid] += 1
        self._seen_hand = set()

    def observe(self, obs: dict, chosen: list) -> None:
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if not opts:
            return
        state = obs.get("current") or {}
        me = state.get("yourIndex")
        if me is None:
            return
        self.selects += 1
        # what is in our hand right now -- the denominator a swap really has
        try:
            hand = (state["players"][me].get("hand") or [])
            for c in hand:
                if c and c.get("id") in self.deck:
                    self._seen_hand.add(c["id"])
        except (KeyError, IndexError, TypeError):
            pass
        picked = set(chosen or [])
        here: set[int] = set()
        for j, o in enumerate(opts):
            try:
                _, cid, _, _ = optfeat.option_features(obs, o)
            except Exception:  # noqa: BLE001 -- one bad option must not kill the audit
                continue
            if not cid or cid not in self.deck:
                continue
            here.add(cid)
            if j in picked:
                self.taken[cid] += 1
        for cid in here:
            self.offered[cid] += 1


def instrument(agent: harness.Agent, tally: Tally) -> harness.Agent:
    """Wrap an agent so every select it makes is tallied.

    The wrapper travels with the AGENT, not with a seat index, which is the
    whole reason this is not vulnerable to rule 18's seat bug: `evaluate_paired`
    alternates seats by swapping the argument order, and the wrapped callable
    goes wherever its agent goes.
    """
    def wrapped(obs: dict):
        out = agent(obs)
        try:
            tally.observe(obs, out)
        except Exception:  # noqa: BLE001 -- never let the audit change the game
            pass
        return out
    return wrapped


def run_anchor(key: str, games: int) -> tuple[Tally, dict]:
    ours, theirs, their_deck = ANCHORS[key]
    _, deck_us = resolve_deck("grimmsnarl")
    their_name, deck_them = resolve_deck(their_deck)
    _, agent_us = build_agent(ours, deck_us, "grimmsnarl")
    # pass the deck name: a rule pilot's identity includes the 60 it is piloting
    # (EVIDENCE §8ax), and this script's whole job is to be a cross-check
    label_them, agent_them = build_agent(theirs, deck_them, their_name)

    tally = Tally(dict(grimmsnarl.DECKLIST))
    wrapped = instrument(agent_us, tally)

    def on_game(match: int, a_seat: int, r: harness.GameResult) -> None:
        tally.end_game()

    t0 = time.monotonic()
    res = harness.evaluate_paired(wrapped, agent_them, deck_us, deck_them,
                                  matches=max(games // 2, 1), on_game=on_game)
    dt = time.monotonic() - t0
    print(f"  {key:<11} vs {label_them:<18} [{their_name}]  "
          f"score={res['score']:.3f} [{res['wilson_low']:.3f}, "
          f"{res['wilson_high']:.3f}] over {res['games']} games, {dt:.0f}s",
          flush=True)
    return tally, res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=300,
                    help="games per anchor (a liveness RATE needs far fewer "
                         "than a win rate; 300 puts a 0.3 plays/game estimate "
                         "at about ±0.06)")
    ap.add_argument("--anchors", nargs="*", default=list(ANCHORS))
    ap.add_argument("--out", default="out/logs/p34_liveness.json")
    args = ap.parse_args()

    deck = dict(grimmsnarl.DECKLIST)
    print(f"\nLIVENESS OF OUR 60, BY MATCHUP -- {args.games} games per anchor\n")
    tallies: dict[str, Tally] = {}
    scores: dict[str, dict] = {}
    for key in args.anchors:
        if key not in ANCHORS:
            raise SystemExit(f"unknown anchor {key!r}; have {list(ANCHORS)}")
        tallies[key], scores[key] = run_anchor(key, args.games)

    keys = list(tallies)
    print("\n⚠ The score column above is a CROSS-CHECK, not a result: at these n")
    print("  it is far too wide to revise §8ap's table.  If one disagrees with")
    print("  §8ap by more than its CI, the harness changed and everything below")
    print("  is suspect (rule 18 -- compute the headline a second way).\n")

    # --- the matrix ----------------------------------------------------------
    print("PLAYS PER GAME, by matchup (taken / games)\n")
    hdr = "%-26s %3s " % ("card", "n") + " ".join("%9s" % k[:9] for k in keys) \
        + " %9s %9s" % ("weighted", "spec")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for cid, n in deck.items():
        pg = {k: tallies[k].taken[cid] / max(tallies[k].games, 1) for k in keys}
        w_tot = sum(SHARES[k] for k in keys)
        weighted = sum(SHARES[k] * pg[k] for k in keys) / w_tot
        mirror = pg.get("mirror")
        # specificity: how much of this card's use the mirror CANNOT see.
        # 1.0 = never played in the mirror but played elsewhere; 0.0 = the
        # mirror sees it at least as often as the field average does.
        spec = 0.0
        if mirror is not None and weighted > 1e-9:
            spec = max(0.0, 1.0 - mirror / weighted)
        rows.append((spec, weighted, cid, n, pg))

    for spec, weighted, cid, n, pg in sorted(rows, key=lambda r: -r[0]):
        flag = ""
        if weighted < 0.30:
            flag = "  under every sizing floor"
        elif spec >= 0.5:
            flag = "  <== MIRROR-BLIND"
        print("%-26s %3d " % (name_of(cid)[:26], n)
              + " ".join("%9.2f" % pg[k] for k in keys)
              + " %9.2f %9.2f%s" % (weighted, spec, flag))

    print("\n  'weighted' = field-share-weighted plays/game over the anchors run")
    print("  'spec'     = 1 - mirror/weighted, clipped at 0.  It is the fraction")
    print("               of a card's use that a MIRROR-ONLY A/B cannot see.")
    print("\n🔴 HOW TO READ THIS FOR A DESIGN (and it is the whole point):")
    print("  spec ~ 0            -> test the swap MIRROR-ONLY.  Cheaper, direct")
    print("                         head-to-head, and unbiased for this card.")
    print("  spec >= 0.5         -> a mirror A/B answers the wrong question by")
    print("                         construction (§8al's Tool Scrapper).  Must be")
    print("                         stratified, at p33's allocation.")
    print("  weighted < 0.30     -> the slot is under every sizing floor this")
    print("                         project has killed a rule at (§8e 0.2, §8ag")
    print("                         0.27, §8ai 0.187).  ⚠ that caps what a swap")
    print("                         can PAY; it does not say the card is bad.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "games_per_anchor": args.games,
        "anchors": {k: {"score": scores[k]["score"], "games": tallies[k].games,
                        "selects": tallies[k].selects,
                        "offered": {str(c): v for c, v in tallies[k].offered.items()},
                        "taken": {str(c): v for c, v in tallies[k].taken.items()},
                        "drawn_games": {str(c): v for c, v in tallies[k].drawn_games.items()}}
                    for k in keys},
        "deck": {str(c): n for c, n in deck.items()},
        "names": {str(c): name_of(c) for c in deck},
    }, indent=1), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
