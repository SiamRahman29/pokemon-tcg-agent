"""The arena: every strength claim in this project runs through here.

    python scripts/arena.py play A B [--matches N] [--deck-a SPEC] [--deck-b SPEC]
    python scripts/arena.py elo

`play` runs seat-swapped paired matches between two agent specs, appends one
JSONL row per game to `out/arena/games.jsonl` (the permanent archive), and
prints A's score with a Wilson 95% interval. `elo` fits Elo ratings over the
whole archive (`rule:iono` anchored at 1000).

Agent specs:

    rule:<name>  (sample rule-based agents: dragapult, iono, abomasnow, lucario)
    rule:v10[,noS][,tb<sec>]  (the public LB 950+ baseline -- our only strong
                 opponent; `noS` disables its MCTS, `tb` sets its per-decision
                 wall-clock budget in seconds, default 1.5)
    random       (uniform legal choice -- the floor to measure against)

A `rule:<name>` agent is bound to the deck it was tuned for, so pass its own
deck: `rule:iono --deck-a iono`, `rule:dragapult --deck-a dragapult_ex`,
`rule:abomasnow --deck-a mega_abomasnow_ex`, `rule:lucario --deck-a mega_lucario_ex`,
`rule:v10 --deck-a lucario_v10`.

`rule:v10`'s MCTS budgets on wall-clock, so gotcha "CPU contention distorts
timed agents" applies to it exactly as it does to `search:` -- prefer `noS` for
comparisons run under varying load.

Deck specs: `sample` (the SDK sample deck), a `decks/` module name (`iono`,
`dragapult_ex`, ...), or a path to a headerless 60-line deck.csv.

🔴 **ARCHIVED IDENTITIES CARRY WHAT CAN CHANGE THE RESULT (day 22, rule 20).**
A rule pilot archives as `rule:<name>@<deck>` -- it is tuned for exactly one 60
and plays any other through a generic fallback, worth **+0.140** in the one case
measured (EVIDENCE §8ax), and running it off its `DECK_MODULE` deck prints a
warning to **stderr**. A policy agent archives as `bc:<tag>#<md5-8 of the
weights>`, because a bare `bc` follows a moving `sa/policy_net.npz` and a `net=`
path can be repointed by a retrain. Names written before day 22 have neither
suffix and pool everything that shared a spec string.
"""
from __future__ import annotations

import argparse
import json
import math
import random as _random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg import config              # noqa: E402
from ptcg.env import harness, sdk    # noqa: E402

ARENA_DIR = ROOT / "out" / "arena"
GAMES_PATH = ARENA_DIR / "games.jsonl"
# 2 (day 22): every row carries `run`, a per-invocation id. Rows written before
# the bump have no `run` key and are all one undifferentiated pool.
SCHEMA = 2


# --- decks --------------------------------------------------------------------

def resolve_deck(spec: str) -> tuple[str, list[int]]:
    """Resolve a deck spec to (stable name, 60-card id list)."""
    if spec == "sample":
        path = config.find_sample_deck()
        if path is None:
            raise SystemExit("no sample deck found under data/")
        rows = path.read_text().split("\n")
        return "sample", [int(rows[i]) for i in range(60)]
    if spec.endswith(".csv"):
        rows = Path(spec).read_text().split("\n")
        return Path(spec).stem, [int(rows[i]) for i in range(60)]
    import importlib

    mod = importlib.import_module(f"decks.{spec}")
    counts: dict[int, int] = mod.DECKLIST
    return spec, [cid for cid, cnt in counts.items() for _ in range(cnt)]


# --- agents -------------------------------------------------------------------

def make_random_agent(deck: list[int], seed: int = 0) -> harness.Agent:
    """Uniform choice among the legal options -- the measurement floor."""
    rng = _random.Random(seed)

    def agent(obs_dict: dict) -> list:
        obs = sdk.api().to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)
        sel = obs.select
        k = min(max(sel.minCount, 1 if sel.maxCount else 0), sel.maxCount,
                len(sel.option))
        return rng.sample(range(len(sel.option)), k)

    return agent


def _flag_num(flags: set[str], prefix: str, cast):
    """Pull `<prefix><number>` out of a spec's flag set (e.g. `w48`, `mc12`)."""
    for f in flags:
        if f.startswith(prefix):
            rest = f[len(prefix):]
            try:
                return cast(rest)
            except ValueError:
                continue
    return None


def build_agent(spec: str, deck: list[int],
                deck_name: str | None = None) -> tuple[str, harness.Agent]:
    """Build (canonical name, agent). The name is what the archive records, so
    the same config always archives under the same identity.

    🔴 `deck_name` IS PART OF A RULE AGENT'S IDENTITY, and leaving it out cost
    two published findings. A `rule:<name>` pilot is tuned for one 60 and plays
    any other through a generic fallback -- `decks/crustle_v1.py` says so in its
    own docstring -- so `rule:crustle` on `crustle_v1` and `rule:crustle` on
    `crustle` are different instruments. Both archived as plain `rule:crustle`,
    and §8an/§8aq attributed a 0.11 swing between them entirely to the pilot's
    bench logic when the deck argument had changed too. See EVIDENCE §8ax.
    """
    kind = spec.split(":", 1)[0]
    if kind == "rule":
        # rule:<deck-name> -- a self-contained sample rule-based agent, bound to
        # the deck it was tuned for (pass the matching --deck so its card
        # counting is correct; see agentkit.rulebased.DECK_MODULE).
        from agentkit.rulebased import make_rule_agent

        parts = spec.split(":", 1)[1].split(",")
        rname = parts[0]
        flags = {p.strip() for p in parts[1:]}
        overrides: dict = {}
        name = f"rule:{rname}"
        if "noS" in flags:
            overrides["USE_SEARCH"] = False
            name += ",noS"
        budget = _flag_num(flags, "tb", float)
        if budget is not None:
            overrides["SEARCH_TIME_BUDGET"] = budget
            name += f",tb{budget:g}"
        # The deck goes in the identity, and a deck that is not the one this
        # pilot was tuned for is called out rather than merely recorded.
        from agentkit.rulebased import DECK_MODULE

        tuned = DECK_MODULE.get(rname)
        if deck_name is not None:
            name += f"@{deck_name}"
            if tuned and deck_name != tuned:
                # ⚠ STDERR, deliberately. Drivers parse stdout for the score
                # line and drop everything else, so a warning printed to stdout
                # is a warning nobody sees -- which is how this defect survived
                # five experiments. p56/p57/p58 echo stderr even on success.
                print(f"⚠ rule:{rname} is tuned for `{tuned}` and is being run "
                      f"on `{deck_name}`.\n  It will play the unshared slots "
                      f"through a generic fallback, so this is a\n  DIFFERENT "
                      f"instrument from rule:{rname}@{tuned} -- do not compare "
                      f"their\n  scores across runs (EVIDENCE 8ax, HANDOFF "
                      f"rule 20).", file=sys.stderr, flush=True)
        return name, make_rule_agent(rname, deck, overrides)
    if kind == "random":
        return "random", make_random_agent(deck)
    if kind == "search":
        # search[:tag][,noP][,noV][,w<N>][,roll][,mc<S>][,nc<S>] -- the sa
        # determinized-search agent. Every flag applies to THIS instance only,
        # so two configs can be A/B'd head-to-head in one process (the SA_*
        # env vars are module-level and would otherwise apply to both sides).
        from sa.agent import SearchAgent

        tag = spec.split(":", 1)[1] if ":" in spec else ""
        parts = tag.split(",")
        flags = {p.strip() for p in parts[1:]}
        no_pnet = True if "noP" in flags else None
        no_vnet = True if "noV" in flags else None
        # w<N>: determinizations per decision (the real compute knob)
        max_worlds = _flag_num(flags, "w", int)
        # roll: rollout to terminal instead of a 1.5-turn horizon + leaf eval.
        # mc/nc: main / minor per-decision time cap in seconds (rollouts need
        # more than the 4.5s default, and we use 3% of the pool today).
        rollout = "roll" in flags
        return ((f"search:{tag}" if tag else "search"),
                SearchAgent(deck, no_pnet=no_pnet, no_vnet=no_vnet,
                            max_worlds=max_worlds, rollout=rollout,
                            main_cap=_flag_num(flags, "mc", float),
                            minor_cap=_flag_num(flags, "nc", float),
                            prior_bonus=_flag_num(flags, "pb", float),
                            main_only="mo" in flags))
    if kind == "bc":
        # bc[:tag][,net=<path>] -- pure behavioral-clone policy agent. `net=`
        # pins this instance to a specific npz, so two candidate policies can
        # play each other head-to-head in one process.
        from sa.bcagent import PolicyAgent

        tag = spec.split(":", 1)[1] if ":" in spec else ""
        net_path = None
        chip = True
        spread = True
        # off by default, matching PolicyAgent: unproven rules are opt-in, so
        # a plain `bc` is always exactly what a submission would ship
        drag = False
        boss = False
        drag_hi = False
        veto = False
        # on by default, matching PolicyAgent: it cleared its own A/B
        wall = True
        # on by default, matching PolicyAgent: it cleared its own A/B
        source = True
        # the 5th Boss's Orders rule: opt-in, unproven
        bossprize = False
        # E11 bench development: opt-in, unproven
        poffin = False
        # E21 Petrel fetch rules: opt-in, unproven
        fstad = False
        fscrap = False
        # B4 turn-level lookahead: opt-in, unproven
        sequencer = False
        seq_k, seq_dets, seq_budget = 8, 4, 0.35
        seq_reply = False
        # E3 near-tie probe: None = off, and the agent is then byte-identical
        # in behaviour to what it was before the probe existed.
        flip_margin = None
        sym_k = 0
        # E17 / §2.7 -- the clock. Opt-in, unproven, and the only bc component
        # that spends real wall-clock time. Defaults are E17's measured design
        # (probe 10, skip win-prob > 0.85, options <= 5, 20 pairs x 3 arms).
        oracle = False
        orc_probe, orc_sel, orc_arms = 10, 20, 3
        orc_wp, orc_maxopt, orc_tau, orc_cap = 0.85, 5, 0.0, 12.0
        orc_maxdev = 0          # E19a: max overrules per GAME, 0 = unlimited
        # E20 -- the value lookahead. Opt-in, unproven. Defaults are the ones
        # frozen in docs/experiments/E20-value-lookahead.md: W=4 worlds, no
        # value trigger, a shape bound of 12 options.
        vlook = False
        vlk_worlds, vlk_maxopt, vlk_cap = 4, 12, 5.0
        vlk_path = None
        # E21: 0.0/0 reproduce E20 exactly; E21 freezes 1.0 and 3.
        vlk_lcb, vlk_arms, vlk_rand, vlk_tau = 0.0, 0, 0.0, 0.0
        for f in tag.split(",")[1:]:
            f = f.strip()
            if f.startswith("net="):
                net_path = f[4:]
            elif f == "noChip":
                # disable the chip-damage targeting override (sa/targeting.py)
                chip = False
            elif f == "noSpread":
                # disable the Munkidori {D}-spread override (sa/targeting.py)
                spread = False
            elif f in ("drag", "noDrag"):
                # Boss's Orders target ranking (sa/targeting.py), default off
                drag = f == "drag"
            elif f in ("boss", "noBoss"):
                # "play Boss's Orders when it buys a KO", default off
                boss = f == "boss"
            elif f in ("dragHi", "noDragHi"):
                # drag the HIGHEST-HP KO-able target instead of the lowest.
                # Only bites with `drag` on -- it reorders drag_target's output.
                drag_hi = f == "dragHi"
            elif f in ("src", "noSrc"):
                # take Adrena-Brain's counters off a source that HAS 3 of
                # them (sa/targeting.counter_source), default ON
                source = f == "src"
            elif f in ("veto", "noVeto"):
                # P5b: DON'T play Boss's Orders when nothing on their bench is
                # KO-able (32.4% of our plays). Default off until it A/Bs.
                veto = f == "veto"
            elif f in ("bossPrize", "noBossPrize"):
                # "attacking beats dragging" veto (sa/targeting.py). Default
                # off until it clears an A/B.
                bossprize = f == "bossPrize"
            elif f in ("seq", "noSeq"):
                # B4: turn-level lookahead (sa/sequencer.py). Default OFF --
                # an experiment until it clears an A/B (EVIDENCE 8m).
                sequencer = f == "seq"
            elif f in ("reply", "noReply"):
                # B4's DESIGN fix (EVIDENCE 8n): score each candidate after the
                # opponent's reply turn instead of at the end of ours. Implies
                # `seq` -- on its own it does nothing.
                seq_reply = f == "reply"
            elif f.startswith("flip"):
                # E3's teacher-free gate: take the OTHER side of every
                # boundary near-tie under this logit margin. `flip0` is a
                # no-op control that still pays the second forward pass.
                flip_margin = float(f[4:])
            elif f.startswith("sym"):
                # R2 (day 27): average the decision over K bench-slot
                # relabellings (sa/symavg.py, EVIDENCE 8bt). `sym1` is the
                # no-op control -- identity relabelling only.
                sym_k = int(f[3:])
            elif f.startswith("sk"):
                seq_k = int(f[2:])          # candidates per select
            elif f.startswith("sd"):
                seq_dets = int(f[2:])       # determinizations per candidate
            elif f.startswith("sb"):
                seq_budget = float(f[2:])   # seconds per select
            elif f in ("wall", "noWall"):
                # The matchup branch: stop applying chip_target when their
                # Active is a damage-prevention wall (Crustle), because the rule
                # measured -0.126 there while paying +0.077 in the mirror.
                # Default ON: 0.663 [0.642, 0.684] vs rule:crustle against
                # 0.559 for unconditional chip_target, n=2000 each, and it
                # cannot fire in the mirror. `noWall` restores the old behaviour.
                wall = f == "wall"
            elif f == "orc":
                oracle = True
            elif f.startswith("op"):
                orc_probe = int(f[2:])      # stage-1 probe rollouts
            elif f.startswith("os"):
                orc_sel = int(f[2:])        # stage-2 pairs per arm
            elif f.startswith("oa"):
                orc_arms = int(f[2:])       # arms (top-k of the net)
            elif f.startswith("ow"):
                orc_wp = float(f[2:])       # skip above this win probability
            elif f.startswith("om"):
                orc_maxopt = int(f[2:])     # the free trigger: option count
            elif f.startswith("ot"):
                orc_tau = float(f[2:])      # min margin to overrule (POST-HOC)
            elif f.startswith("oc"):
                orc_cap = float(f[2:])      # per-decision seconds cap
            elif f.startswith("od"):
                orc_maxdev = int(f[2:])     # E19a: max overrules per game
            elif f == "vlp":
                # E20: one-ply lookahead scored by the learned value net.
                vlook = True
            elif f.startswith("vnet="):
                vlk_path = f[5:]            # WHICH value net (rule 20)
            elif f.startswith("vw"):
                vlk_worlds = int(f[2:])     # determinized worlds per option
            elif f.startswith("vo"):
                vlk_maxopt = int(f[2:])     # shape bound, not a value trigger
            elif f.startswith("vlcb"):
                vlk_lcb = float(f[4:])      # E21: K in mean - K*sd
            elif f.startswith("varm"):
                vlk_arms = int(f[4:])       # E21: coverage, net's top-k
            elif f.startswith("vtau"):
                vlk_tau = float(f[4:])      # E25: min V-gap to overrule
            elif f.startswith("vrnd"):
                vlk_rand = float(f[4:])     # E22 audit: rate-matched coin flip
            elif f.startswith("vc"):
                vlk_cap = float(f[2:])      # per-decision seconds cap
            elif f in ("fstad", "noFstad"):
                # E21: fetch Spikemuth Gym when no Stadium is ours (0.461
                # firings/game). Default OFF until its own A/B clears the bar.
                fstad = f == "fstad"
            elif f in ("fscrap", "noFscrap"):
                # E21: fetch Tool Scrapper only when a Tool is on THEIR board
                # (0.171 firings/game). Default OFF, same discipline.
                fscrap = f == "fscrap"
            elif f in ("poffin", "noPoffin"):
                # E11: play Buddy-Buddy Poffin when the bench has >=2 free
                # slots. Default OFF until its own A/B clears the bar
                # (docs/experiments/E11-poffin.md).
                poffin = f == "poffin"
            else:
                # The FIRST token of a bc tag is a free-text label
                # (`bc:old,noChip`), so a flag typed as `bc:veto` lands in the
                # label slot and is silently ignored -- that cost one wasted
                # run. Everything after it must be a real flag.
                raise SystemExit(
                    f"unknown bc flag {f!r} in {spec!r}. Note the first token "
                    f"after `bc:` is a LABEL, not a flag -- write "
                    f"`bc:<label>,{f}`.")
        # `net=a.npz+b.npz` is an ensemble spec; every member must exist.
        for _p in (net_path.split("+") if net_path else []):
            if _p and not Path(_p).exists():
                raise SystemExit(f"bc net not found: {_p}")
        try:
            agent = PolicyAgent(deck, net_path, chip_targeting=chip,
                                energy_spread=spread, drag_target=drag,
                                boss_converts=boss, drag_high_hp=drag_hi,
                                boss_veto=veto, counter_source=source,
                                chip_wall_defer=wall, boss_prize_veto=bossprize,
                                sequencer=sequencer,
                                seq_k=seq_k, seq_dets=seq_dets,
                                seq_budget=seq_budget, seq_reply=seq_reply,
                                flip_margin=flip_margin,
                                poffin_force=poffin, sym_k=sym_k,
                                fetch_stadium=fstad, fetch_scrapper=fscrap,
                                oracle=oracle, orc_probe=orc_probe,
                                orc_sel=orc_sel, orc_arms=orc_arms,
                                orc_wp=orc_wp, orc_maxopt=orc_maxopt,
                                orc_tau=orc_tau, orc_cap=orc_cap,
                                orc_maxdev=orc_maxdev,
                                vlook=vlook, vlk_worlds=vlk_worlds,
                                vlk_maxopt=vlk_maxopt, vlk_cap=vlk_cap,
                                vlk_path=vlk_path, vlk_lcb=vlk_lcb,
                                vlk_arms=vlk_arms, vlk_rand=vlk_rand,
                                vlk_tau=vlk_tau)
        except ValueError as exc:
            # the `net=` guard (sa/bcagent.py): a net that exists but fails
            # policynet.load used to fall through to the tracked singleton and
            # score silently under the requested net's name.
            raise SystemExit(f"{spec}: {exc}")
        # 🔴 The VALUE net's bytes go in the identity too. A `vnet=` path is no
        # more an identity than a `net=` path is (rule 20): retraining V and
        # re-running would otherwise pool two different agents under one name.
        nm = (f"bc:{tag}" if tag else "bc") + _net_fp(net_path)
        if vlook:
            # `_net_fp(None)` falls back to the POLICY net's path, which would
            # archive a value-net identity that is not the value net. Demand it.
            if not vlk_path:
                raise SystemExit("bc:...,vlp requires vnet=<path> so the value "
                                 "net's bytes enter the archived identity")
            nm += "/v" + _net_fp(vlk_path)[1:]
        return nm, agent
    raise SystemExit(f"unknown agent spec: {spec!r}")


def _net_fp(net_path: str | None) -> str:
    """`#<8 hex>` of the weights this agent will actually use.

    🔴 RULE 19 APPLIES TO OUR OWN AGENT, NOT ONLY TO ANCHORS. A bare `bc` spec
    passes no `net=`, so it plays whatever `sa/policy_net.npz` happens to be at
    that moment -- and `out/arena/games.jsonl` holds **1,226 games archived
    under the single identity `bc`, spanning 07-28 to 07-31**, across which that
    file was a moving target. `arena.py elo` pools them. Rule 19 was written
    about `rule:<name>` anchors and this is the same defect one seat over.

    ⚠ A `net=` PATH IS NOT AN IDENTITY EITHER: `out/policy_v5.npz` can be
    overwritten by a retrain and every archived row still says `policy_v5.npz`.
    The fingerprint is of the BYTES, so a re-trained net archives as a new
    agent rather than pooling with its predecessor -- which is the behaviour
    §8aw's "rebuild the corpus and a net's map is stale" warning needs.
    """
    import hashlib

    from sa import policynet

    # An ensemble's identity is ALL of its members, in order: swapping one
    # member is a different agent, and voting order is part of the spec
    # (member 0 supplies the count rule).
    if net_path and "+" in net_path:
        h = hashlib.md5()
        for p in net_path.split("+"):
            if not p:
                continue
            q = Path(p)
            if not q.exists():
                return "#none"
            h.update(q.read_bytes())
        return "#" + h.hexdigest()[:8]
    path = Path(net_path) if net_path else Path(policynet._PATH)
    if not path.exists():
        return "#none"
    return "#" + hashlib.md5(path.read_bytes()).hexdigest()[:8]


# --- play ---------------------------------------------------------------------

def cmd_play(args: argparse.Namespace) -> int:
    if config.find_sdk_dir() is None:
        print("cg engine not found (paste the sample submission into data/).")
        return 1
    sdk.load()

    deck_name_a, deck_a = resolve_deck(args.deck_a)
    deck_name_b, deck_b = resolve_deck(args.deck_b)
    name_a, agent_a = build_agent(args.a, deck_a, deck_name_a)
    name_b, agent_b = build_agent(args.b, deck_b, deck_name_b)
    # zero the health counters so the line printed at the end describes THIS
    # run and not whatever else the process did first
    if "sa.bcagent" in sys.modules:
        sys.modules["sa.bcagent"].reset_stats()

    games_path = Path(args.archive) if args.archive else GAMES_PATH
    games_path.parent.mkdir(parents=True, exist_ok=True)

    # 🔴 ARCHIVES APPEND, AND A RE-RUN USED TO BE INVISIBLE ONCE WRITTEN. Every
    # driver writes many cells into one file, so refusing to append would break
    # the normal pattern -- but nothing distinguished "the second cell of this
    # experiment" from "the same cell run twice". `out/arena/p57_e8.jsonl`
    # carries 3,000 games per v5c CONTROL cell against 1,500 per treatment cell,
    # because the control was re-run for the v7pad pass into the same file. The
    # published numbers are safe (drivers parse the score line arena prints),
    # but anyone re-deriving from the archive gets a control pooled over two
    # runs that was never the published control. So: stamp the run, and say so
    # out loud when the target already holds this exact cell.
    run_id = f"{int(time.time()):x}-{_random.Random().getrandbits(24):06x}"
    dup = 0
    if games_path.exists():
        cell = {(name_a, name_b, deck_name_a, deck_name_b),
                (name_b, name_a, deck_name_b, deck_name_a)}
        for line in games_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if (r.get("agent0"), r.get("agent1"),
                        r.get("deck0"), r.get("deck1")) in cell:
                    dup += 1
    if dup:
        print(f"⚠ {games_path.name} ALREADY HOLDS {dup} games of this exact "
              f"cell.\n  Appending. This run is tagged run={run_id}; anything "
              f"re-derived from this\n  file without splitting on `run` pools "
              f"two separate measurements.", file=sys.stderr, flush=True)

    rows: list[dict] = []
    t_start = time.monotonic()
    # Append+flush per game. Buffering rows until the end meant a killed run
    # lost every game it had played, which made long runs un-abortable.
    archive_fh = games_path.open("a", encoding="utf-8")

    def on_game(match: int, a_seat: int, r: harness.GameResult) -> None:
        names = (name_a, name_b) if a_seat == 0 else (name_b, name_a)
        decks = ((deck_name_a, deck_name_b) if a_seat == 0
                 else (deck_name_b, deck_name_a))
        row = {
            "schema": SCHEMA, "ts": time.time(), "run": run_id, "match": match,
            "agent0": names[0], "agent1": names[1],
            "deck0": decks[0], "deck1": decks[1],
            "winner": r.winner, "turns": r.turns, "selects": r.selects,
            "lat0": harness.latency_summary(r.decision_ms[0]),
            "lat1": harness.latency_summary(r.decision_ms[1]),
            # seat-indexed, like agent0/agent1 above
            "pool0": round(r.pool_left[0], 1),
            "pool1": round(r.pool_left[1], 1),
        }
        rows.append(row)
        archive_fh.write(json.dumps(row) + "\n")
        archive_fh.flush()
        print(f"  match {match} seat{a_seat}: winner={r.winner} "
              f"turns={r.turns} selects={r.selects}", flush=True)

    print(f"{name_a} [{deck_name_a}] vs {name_b} [{deck_name_b}], "
          f"{args.matches} paired matches ({2 * args.matches} games)...")
    try:
        res = harness.evaluate_paired(agent_a, agent_b, deck_a, deck_b,
                                      matches=args.matches, on_game=on_game)
    finally:
        # rows are already on disk (on_game flushes each one); just close.
        # Do NOT re-write `rows` here -- that double-counted every game.
        archive_fh.close()

    dt = time.monotonic() - t_start
    print(f"\nA={name_a}: score={res['score']:.3f} "
          f"[{res['wilson_low']:.3f}, {res['wilson_high']:.3f}] "
          f"W{res['wins']}/D{res['draws']}/L{res['losses']} over {res['games']} games")
    print(f"  as P0: W/D/L={res['a_as_p0_wdl']}  as P1: W/D/L={res['a_as_p1_wdl']}")
    print(f"  elapsed {dt:.1f}s; archived {len(rows)} rows -> {games_path}")
    # Kaggle converts an exhausted thinking pool into a LOSS; this harness does
    # not, so a slow agent can win here and time out on the ladder.
    worst: dict[str, float] = {}
    for r in rows:  # agents swap seats every game, so key by name not seat
        for seat in (0, 1):
            name = r[f"agent{seat}"]
            worst[name] = min(worst.get(name, 600.0), r[f"pool{seat}"])
    for name, left in worst.items():
        if left < 300.0:
            flag = "  <-- WOULD TIME OUT ON KAGGLE" if left <= 0 else ""
            print(f"  pool left ({name}): min {left:.0f}s of 600s{flag}")
    # 🔴 THE DEGRADATION READOUT. `bcagent.__call__` wraps every decision in a
    # catch-all that returns `range(minCount)` -- index order -- and prints a
    # traceback to stderr. An arm running that on EVERY decision still returns
    # legal moves, still finishes its games, and still prints a perfectly
    # ordinary score line, which is what the drivers parse. Day 15 built the
    # counters to make it a direct read and then wired them only into the
    # SUBMISSION (`build_submission.py`), so the ladder had the instrument and
    # the arena -- the one day 17 calls "the ONLY instrument" -- did not.
    # ⚠ STATS is process-global, so on an in-process A/B this is the sum over
    # BOTH sides and cannot say which arm degraded. Non-zero means re-run the
    # arms separately, not that the treatment is at fault.
    if "sa.bcagent" in sys.modules:
        print(f"  {sys.modules['sa.bcagent'].health_line()}")
    # And the per-component detail, for any agent carrying a sequencer. ⚠ BOTH
    # blocks are kept deliberately (day-22 merge): `[health]` says whether the
    # AGENT degraded, this says whether the PLANNER did, and E5 needed the
    # second to defend its confirm cell (`errors: 0`, `budget_aborts: 0` in all
    # four arms). EVIDENCE §8bb -- and the counters belong in a manifest as well
    # as on stdout, because §8bb was nearly filed as unauditable when a later
    # pass looked for them in `out/logs/` and found nothing.
    for name, agent in ((name_a, agent_a), (name_b, agent_b)):
        seq = getattr(agent, "seq", None)
        if seq is not None:
            st = seq.stats
            per_plan = st["sim_s"] / max(st["planned"], 1)
            print(f"  sequencer ({name}): {st['planned']} completed, "
                  f"{st['overruled']} overrules, {st['fellback']} errors, "
                  f"{st['aborted_budget']} budget aborts, "
                  f"{st['sim_s']:.1f}s total ({per_plan:.3f}s/completed)")
    if "sa.planner" in sys.modules:  # search internals, summed over both sides
        st = sys.modules["sa.planner"].STATS
        if st["decides"]:
            print(f"  planner: {st['decides']} decides, "
                  f"{st['worlds'] / st['decides']:.1f} worlds/decide, "
                  f"{100 * st['spent_s'] / max(st['budget_s'], 1e-9):.0f}% of "
                  f"budget used, {st['spent_s']:.0f}s total")
        if st["rollouts"]:
            print(f"  rollouts: {st['rollouts']} "
                  f"({100 * st['roll_terminal'] / st['rollouts']:.0f}% reached "
                  f"terminal), {st['roll_steps'] / st['rollouts']:.0f} steps each")
        if st["anchored"]:
            print(f"  search overruled the clone on "
                  f"{100 * st['deviated'] / st['anchored']:.0f}% of anchored "
                  f"decisions ({st['deviated']}/{st['anchored']})")
    return 0


# --- elo ----------------------------------------------------------------------

def _load_rows(path: Path = GAMES_PATH) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


_LN10_400 = math.log(10.0) / 400.0
# Damping on the diagonal-Newton step. Below 1.0 because the Hessian is not
# actually diagonal -- players are coupled through their shared games -- and an
# undamped diagonal step can overshoot when two agents play mostly each other.
_DAMP = 0.8
# Two virtual draws against a phantom at `anchor_rating`. Bradley-Terry has no
# finite maximum for an unbeaten player, so without this a 20-0 agent's rating
# runs away and prints whatever the iteration cap happened to reach. At two
# games this is under 0.2% of a 2,000-game player's evidence and invisible;
# on a 20-game player it is the only thing keeping the number finite.
_PRIOR_N = 2.0


def fit_elo(rows: list[dict], anchor: str = "rule:iono",
            anchor_rating: float = 1000.0, iters: int = 2000,
            lr: float = _DAMP) -> tuple[dict[str, float], float]:
    """Batch-gradient Bradley-Terry fit (draw = half a win), all games weighted
    equally regardless of when they were played. Anchored so ratings are
    comparable across refits as the archive grows. Returns (ratings, max final
    step in Elo points) -- see `cmd_elo`, which refuses to print an unconverged
    fit.

    🔴 **THE STEP MUST BE SCALED BY THE CURVATURE, and this is rule 15.**
    Until day 22 this function took a FIXED `lr=8.0` step on an UNNORMALISED
    batch gradient. The gradient sums over a player's games, so its curvature
    grows with n while the step did not: past roughly 175 games per player the
    iteration is divergent, and it oscillates instead of converging. On the real
    `games.jsonl` that meant `rule:crustle` (1,320 games) swinging **8,586 Elo**
    between consecutive iterations and reading -3632 / +258 / +3397 / -3275 at
    iterations 499 / 500 / 501 / 502. **Every rating this printed was an
    arbitrary sample of an oscillation, including small ones** -- even 30-game
    anchors swung 200+. Nothing published rests on it (every Elo figure in
    `EVIDENCE` is a win-rate conversion), which is exactly why it survived
    fifteen days: an unused instrument is never checked against reality.

    ⚠ Dividing by the game COUNT alone is not enough either -- it is stable, but
    a near-ceiling player's curvature is p(1-p) << 0.25 and 500 iterations left
    a 1.3 Elo residual. The step is the diagonal Newton one: gradient over the
    summed per-game curvature p(1-p)*ln(10)/400, damped by `_DAMP`.
    """
    games = [(r["agent0"], r["agent1"], r["winner"]) for r in rows]
    players = sorted({p for a0, a1, _ in games for p in (a0, a1)})
    rating = {p: anchor_rating for p in players}
    step = 0.0
    for _ in range(iters):
        grad = {p: 0.0 for p in players}
        hess = {p: 0.0 for p in players}   # -d2(loglik)/d(rating)^2, diagonal
        for a0, a1, winner in games:
            s0 = 1.0 if winner == 0 else 0.5 if winner == 2 else 0.0
            e0 = 1.0 / (1.0 + 10.0 ** ((rating[a1] - rating[a0]) / 400.0))
            grad[a0] += s0 - e0
            grad[a1] += (1.0 - s0) - (1.0 - e0)
            h = e0 * (1.0 - e0) * _LN10_400
            hess[a0] += h
            hess[a1] += h
        step = 0.0
        for p in players:
            # the prior's own gradient and curvature: _PRIOR_N virtual draws
            # against a phantom fixed at `anchor_rating`
            ep = 1.0 / (1.0 + 10.0 ** ((anchor_rating - rating[p]) / 400.0))
            g = grad[p] + _PRIOR_N * (0.5 - ep)
            h = hess[p] + _PRIOR_N * ep * (1.0 - ep) * _LN10_400
            d = lr * g / h if h > 0.0 else 0.0
            rating[p] += d
            step = max(step, abs(d))
        if step < 1e-4:
            break
    # ⚠ Re-anchor ONCE, after convergence, not every pass. Re-anchoring inside
    # the loop shifts every player by the anchor's own residual each iteration,
    # which for a component that never played the anchor is pure injected drift
    # that only the 2-game prior pushes back against -- 500 passes still left a
    # 2.0 Elo residual. The shift is a constant offset and commutes with the
    # fit, so doing it last costs nothing and the iteration becomes plain
    # Newton ascent on a concave objective.
    if anchor in rating:
        shift = anchor_rating - rating[anchor]
        for p in players:
            rating[p] += shift
    return rating, step


def anchor_component(rows: list[dict], anchor: str = "rule:iono") -> set[str]:
    """Players joined to `anchor` by a chain of games -- the only ones whose
    rating is on the anchor's scale at all.

    🔴 Two agents that played only each other form their own component: their
    DIFFERENCE is identified, their LEVEL is not, and the fit pins it with
    nothing but the prior. Printing them in one sorted column next to anchored
    ratings invites exactly the comparison the data cannot support, so `elo`
    marks them.
    """
    adj: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        a0, a1 = r["agent0"], r["agent1"]
        adj[a0].add(a1)
        adj[a1].add(a0)
    if anchor not in adj:
        return set()
    seen, stack = {anchor}, [anchor]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def cmd_elo(args: argparse.Namespace) -> int:
    path = Path(args.archive) if args.archive else GAMES_PATH
    rows = _load_rows(path)
    if not rows:
        print(f"no archive at {path}; run `arena.py play` first.")
        return 1
    counts: Counter = Counter()
    lat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for seat in (0, 1):
            name = r[f"agent{seat}"]
            counts[name] += 1
            ls = r.get(f"lat{seat}") or {}
            if ls.get("n"):
                lat[name].append(ls["p99"])
    ratings, step = fit_elo(rows)
    # Rule 9: a metric that never prints is not a metric that passed. The fit
    # silently diverged for fifteen days; the convergence residual is now a
    # required read, and an unconverged fit refuses rather than printing.
    if step > 0.5:
        print(f"🔴 THE FIT DID NOT CONVERGE: final step {step:.1f} Elo. "
              f"Do not quote these ratings.")
        return 1
    linked = anchor_component(rows)
    print(f"Elo over {len(rows)} archived games (rule:iono anchored at 1000; "
          f"converged, final step {step:.5f} Elo):\n")
    for name, elo in sorted(ratings.items(), key=lambda kv: -kv[1]):
        mark = "  " if name in linked else "🔴"
        line = f"{mark}{elo:7.1f}  {name}  ({counts[name]} games)"
        if lat[name]:
            line += f"  p99={sum(lat[name]) / len(lat[name]):.0f}ms"
        print(line)
    loose = sorted(set(ratings) - linked)
    if loose:
        print(f"\n🔴 {len(loose)} agents (marked) never played a game connected "
              "to the anchor.\n  Their LEVEL is set by the prior, not by "
              "evidence -- only their difference\n  from others in their own "
              "component means anything. Do not read them off\n  this column.")
    print("\n⚠ Ratings pool the WHOLE archive regardless of when a game was "
          "played. An\n  agent identity that meant different things on "
          "different days (rule 19) is\n  averaged, not separated. This is a "
          "browsing tool; strength claims come from\n  `play`'s back-to-back "
          "score line.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="run paired matches and archive them")
    p.add_argument("a", help="agent spec for A")
    p.add_argument("b", help="agent spec for B")
    p.add_argument("--matches", type=int, default=10,
                   help="paired matches (2 games each; default 10)")
    p.add_argument("--deck-a", default="sample")
    p.add_argument("--deck-b", default="sample")
    p.add_argument("--archive", default=None,
                   help="archive path (default: out/arena/games.jsonl)")
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("elo", help="fit Elo over the archive")
    p.add_argument("--archive", default=None,
                   help="archive path (default: out/arena/games.jsonl)")
    p.set_defaults(fn=cmd_elo)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
