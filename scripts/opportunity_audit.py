"""Does the agent take the plays its deck is built around?

Watching replays turns up specific missed lines ("it had Rare Candy and an
Impidimp and evolved the slow way"). This turns that anecdote into a rate: for
each tracked opportunity, how many MAIN selects offered it, and how often did we
take it. An opportunity offered 200 times and taken 12 times is a bug worth a
rule; one offered twice is noise.

    python scripts/opportunity_audit.py --matches 100
    python scripts/opportunity_audit.py --agent "bc" --opponent "rule:v10,noS" \
        --deck grimmsnarl --deck-b lucario_v10 --matches 200

`taken` counts only the *first* returned index for MAIN selects, because MAIN
has maxCount == 1 -- the agent picks exactly one action, so an option that is
merely ranked second was not played.

A rate is only meaningful next to the demonstrators' rate for the same option,
because these options persist across every select of a turn -- "Rare Candy was
legal and I played a supporter first" is correct play, not a miss. So:

    python scripts/opportunity_audit.py --corpus artifacts/pds_v2

reads the same opportunity set straight out of the training shards (which store
the option type one-hot, card id and target id per option, plus what the player
actually chose) and prints the top-400-players' take-rate. Compare the two
tables; a gap is a bug, a match is not.

**Per-turn binary is not enough for repeatable lines.** Adrena-Brain is once
*per Pokemon*, so with two Munkidori a turn offers two activations; a per-turn
`any()` scores a single use as 100% and cannot see the missed second one. Each
tracked line therefore declares its multiplicity in `MULTIPLICITY`:

    "turn"  -- at most one per turn no matter how many copies (Supporters,
               Stadiums, the 1-per-turn manual energy attach). Per-turn binary
               is the honest number and "count" would just inflate it.
    "count" -- repeatable within a turn. The denominator is the number of
               distinct offers in the turn, not 1.

Only `munkidori_adrena_brain` is "count" today, because it is the only tracked
line whose copies are separately countable **on both sides**: one ABILITY option
per Munkidori, live and in the shards alike. Items are repeatable too, but a
Rare Candy option carries no target (the target is a follow-up select), so
2 copies in hand and 1 Impidimp would read as 2 opportunities -- an invented
denominator. Leave those at "turn" until there is a way to count real targets.

The `opps`/`rate` columns use that. `turns` stays for comparison with the old,
blinder numbers; for a "turn" line the two are identical by construction.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402

sdk.load()

from cg.api import (AreaType, OptionType, Pokemon, SelectContext,  # noqa: E402
                    to_observation_class)
from ptcg.env import harness  # noqa: E402

import arena  # noqa: E402

# --- the grimmsnarl deck's core lines ------------------------------------
RARE_CANDY = 1079
MUNKIDORI = 112
DARK_ENERGY = 7
IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL_EX = 648
FROSLASS = 104
BOSS_ORDERS = 1182       # switch in an opponent's benched Pokemon
SPIKEMUTH_GYM = 1259     # stadium: tutor a Marnie's Pokemon each turn
TOOL_SCRAPPER = 1137     # discard up to 2 tools
PETREL = 1219            # tutor ANY trainer -- effectively extra Boss's Orders

# cards whose bare PLAY rate we want, card id -> audit name
PLAY_CARDS = {
    BOSS_ORDERS: "boss_orders_play",
    SPIKEMUTH_GYM: "spikemuth_gym_play",
    TOOL_SCRAPPER: "tool_scrapper_play",
    PETREL: "petrel_play",
    RARE_CANDY: "rare_candy_play",
}
# Crustle's Mysterious Rock Inn prevents all attack damage from ex Pokemon.
IMMUNE_TO_EX = {345}

# How many times a line can legitimately be taken in one turn. See the module
# docstring -- getting this wrong is what made `munkidori_adrena_brain` read
# 99.3% when a second activation was being missed on most two-Munkidori turns.
MULTIPLICITY = {
    "boss_orders_play": "turn",        # Supporter: 1 per turn
    "petrel_play": "turn",             # Supporter: 1 per turn
    "spikemuth_gym_play": "turn",      # Stadium: 1 per turn
    "tool_scrapper_play": "turn",      # Item, repeatable -- but see docstring
    "rare_candy_play": "turn",         # Item, repeatable -- but see docstring
    "munkidori_adrena_brain": "count",  # once PER POKEMON (engine-verified)
    "dark_energy_to_munkidori": "turn",  # 1 manual energy attach per turn
    "evolve_impidimp_to_morgrem": "turn",  # per Pokemon, but targets uncountable
    "attack_into_ex_immune_active": "turn",
}


def multiplicity(kind: str) -> str:
    return MULTIPLICITY.get(kind, "turn")


def _card(obs, area: int, index: int, player_index: int):
    """Resolve the card an option points at (V10's get_card, verbatim logic)."""
    player = obs.current.players[player_index]
    try:
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return player.hand[index]
        if area == AreaType.DISCARD:
            return player.discard[index]
        if area == AreaType.ACTIVE:
            return player.active[index]
        if area == AreaType.BENCH:
            return player.bench[index]
        if area == AreaType.PRIZE:
            return player.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except (IndexError, TypeError):
        return None
    return None


def _cid(card) -> int:
    return getattr(card, "id", getattr(card, "cardId", -1))


def classify(obs, option) -> str | None:
    """Which tracked opportunity, if any, does this option represent?"""
    me = obs.current.yourIndex
    opp = obs.current.players[1 - me]

    if option.type == OptionType.PLAY:
        card = _card(obs, AreaType.HAND, option.index, me)
        if card is not None and _cid(card) in PLAY_CARDS:
            return PLAY_CARDS[_cid(card)]

    if option.type == OptionType.ABILITY:
        card = _card(obs, option.area, option.index, me)
        if card is not None and _cid(card) == MUNKIDORI:
            return "munkidori_adrena_brain"

    if option.type == OptionType.ATTACH:
        card = _card(obs, AreaType.HAND, option.index, me)
        target = _card(obs, option.inPlayArea, option.inPlayIndex, me)
        if (card is not None and _cid(card) == DARK_ENERGY
                and isinstance(target, Pokemon) and _cid(target) == MUNKIDORI):
            return "dark_energy_to_munkidori"

    if option.type == OptionType.EVOLVE:
        card = _card(obs, AreaType.HAND, option.index, me)
        target = _card(obs, option.inPlayArea, option.inPlayIndex, me)
        if card is not None and isinstance(target, Pokemon):
            if _cid(card) == MORGREM and _cid(target) == IMPIDIMP:
                return "evolve_impidimp_to_morgrem"

    if option.type == OptionType.ATTACK:
        active = opp.active[0] if opp.active else None
        if active is not None and _cid(active) in IMMUNE_TO_EX:
            return "attack_into_ex_immune_active"

    return None


# Contexts where we point an effect at one of the opponent's Pokemon:
# Shadow Bullet's 30-damage bench snipe, Adrena-Brain's counter move, and
# Boss's Orders' drag. "Finish the cheap one" should show up here as a bias
# toward the lowest-HP option.
TARGET_CONTEXTS = {
    int(SelectContext.DAMAGE): "damage_target",              # Shadow Bullet
    int(SelectContext.DAMAGE_COUNTER): "counter_target",     # Adrena-Brain
    int(SelectContext.DAMAGE_COUNTER_ANY): "counter_target",
    int(SelectContext.TO_ACTIVE): "drag_target",             # Boss's Orders
}


def target_choice(obs, picked: list) -> tuple[str, bool] | None:
    """(kind, chose_lowest_hp) when a select points at >=2 opponent Pokemon."""
    kind = TARGET_CONTEXTS.get(int(obs.select.context))
    if kind is None:
        return None
    me = obs.current.yourIndex
    hps: dict[int, int] = {}
    for i, option in enumerate(obs.select.option):
        if option.playerIndex == me:
            continue
        card = _card(obs, option.area, option.index, option.playerIndex)
        if isinstance(card, Pokemon):
            hps[i] = card.hp
    if len(hps) < 2:
        return None
    chosen = [i for i in list(picked)[:1] if i in hps]
    if not chosen:
        return None
    return kind, hps[chosen[0]] == min(hps.values())


def bump(opp_avail: dict, opp_taken: dict, kind: str, key, n_offers: int,
         took: bool) -> None:
    """Accumulate per-turn offers/uses for one tracked line.

    `n_offers` is how many distinct instances of the line this select showed;
    a turn's denominator is the largest that ever got to 1 at once. For a
    "turn" line it is pinned to 1, which reproduces the old binary exactly."""
    if multiplicity(kind) != "count":
        n_offers = 1
    a = opp_avail.setdefault(kind, {})
    a[key] = max(a.get(key, 0), n_offers)
    if took:
        t = opp_taken.setdefault(kind, {})
        t[key] = t.get(key, 0) + 1


DARK_TYPE = 7  # pk.energies holds energy *types*; Basic {D} Energy is type 7


def _dark(pk) -> int:
    return sum(1 for e in (getattr(pk, "energies", None) or [])
               if e == DARK_TYPE)


def energy_spread(obs, picked: list) -> str | None:
    """Was a {D} attach a spread choice, and did we take it?

    Adrena-Brain is once *per Pokemon* and its energy condition is a threshold,
    not a cost (both engine-verified): a 2nd {D} on a Munkidori that already has
    one buys literally nothing, while the 1st {D} on a bare Munkidori buys a
    whole extra activation -- 3 more damage counters moved. Munkidori is *not*
    a Marnie's Pokemon, so Punk Up cannot attach to it and this hand attach is
    the only source, one per turn.

    Fires only when the select offers both a bare and an already-energised
    Munkidori, i.e. when the mistake is actually available to make."""
    me = obs.current.yourIndex
    bare, loaded = [], []
    for i, option in enumerate(obs.select.option):
        if option.type != OptionType.ATTACH:
            continue
        card = _card(obs, AreaType.HAND, option.index, me)
        target = _card(obs, option.inPlayArea, option.inPlayIndex, me)
        if card is None or _cid(card) != DARK_ENERGY:
            continue
        if not isinstance(target, Pokemon) or _cid(target) != MUNKIDORI:
            continue
        (bare if _dark(target) == 0 else loaded).append(i)
    if not bare or not loaded:
        return None
    chosen = set(list(picked)[:1])
    if chosen & set(bare):
        return "spread (correct)"
    if chosen & set(loaded):
        return "stacked (wasted)"
    return "attached elsewhere / other play"


def wrap(agent, avail: Counter, taken: Counter, turn_avail: dict,
         turn_taken: dict, game: list, tgt_n: Counter = None,
         tgt_low: Counter = None, opp_avail: dict = None,
         opp_taken: dict = None, spread: Counter = None):
    """Count tracked opportunities offered to, and used by, `agent`."""

    def wrapped(obs_dict):
        picked = agent(obs_dict)
        try:
            if obs_dict.get("select") is None:
                return picked
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return picked
            if tgt_n is not None:
                hit = target_choice(obs, picked)
                if hit is not None:
                    tgt_n[hit[0]] += 1
                    tgt_low[hit[0]] += hit[1]
            if obs.select.context != SelectContext.MAIN:
                return picked
            if spread is not None:
                hit = energy_spread(obs, picked)
                if hit is not None:
                    spread[hit] += 1
            kinds: dict[str, list[int]] = {}
            for i, option in enumerate(obs.select.option):
                kind = classify(obs, option)
                if kind is not None:
                    kinds.setdefault(kind, []).append(i)
            if not kinds:
                return picked
            chosen = list(picked)[:1]  # MAIN is maxCount == 1
            key_turn = (game[0], obs.current.turn)
            for kind, idxs in kinds.items():
                avail[kind] += 1
                turn_avail.setdefault(kind, set()).add(key_turn)
                took = any(i in chosen for i in idxs)
                if took:
                    taken[kind] += 1
                    turn_taken.setdefault(kind, set()).add(key_turn)
                if opp_avail is not None:
                    bump(opp_avail, opp_taken, kind, key_turn, len(idxs), took)
        except Exception:  # never let the audit break a game
            pass
        return picked

    return wrapped


# --- the same opportunity set, read out of the training shards -----------
# optfeat.option_features writes a type one-hot into opt_dense[0:17], the
# resolved card id into opt_card and the in-play target into opt_target.
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10

# name -> (option type, card id, target card id or None)
CORPUS_RULES = {
    "munkidori_adrena_brain": (OPT_ABILITY, MUNKIDORI, None),
    "dark_energy_to_munkidori": (OPT_ATTACH, DARK_ENERGY, MUNKIDORI),
    "evolve_impidimp_to_morgrem": (OPT_EVOLVE, MORGREM, IMPIDIMP),
    **{name: (OPT_PLAY, cid, None) for cid, name in PLAY_CARDS.items()},
}


def audit_corpus(ds: str) -> int:
    """Demonstrator take-rate per select, from the policy-dataset shards."""
    import numpy as np

    paths = sorted((ROOT / ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / ds}")

    avail: Counter = Counter()
    taken: Counter = Counter()
    turn_avail: dict[str, set] = {k: set() for k in CORPUS_RULES}
    turn_taken: dict[str, set] = {k: set() for k in CORPUS_RULES}
    opp_avail: dict = {}
    opp_taken: dict = {}
    n_selects = 0
    for path in paths:
        z = np.load(path)
        odense, card = z["opt_dense"], z["opt_card"]
        target = z["opt_target"] if "opt_target" in z else np.zeros_like(card)
        chosen, off = z["opt_chosen"], z["opt_off"]
        gid = z["gid"]
        # features.py writes dense[0] = min(turn, 40) / 40
        turn = np.rint(z["dense"][:, 0] * 40.0).astype(int)
        n_selects += len(off) - 1
        for kind, (otype, cid, tid) in CORPUS_RULES.items():
            hit = (odense[:, otype] == 1.0) & (card == cid)
            if tid is not None:
                hit &= target == tid
            if not hit.any():
                continue
            # collapse per option -> per select via the option offsets
            rows = np.searchsorted(off, np.flatnonzero(hit), side="right") - 1
            picked = chosen[hit] > 0.5
            # how many matching options this select showed -- the per-select
            # multiplicity a per-turn binary throws away
            per_select = Counter(rows.tolist())
            for row, was in zip(rows, picked):
                avail[kind] += 1
                key = (path.name, int(gid[row]), int(turn[row]))
                turn_avail[kind].add(key)
                if was:
                    taken[kind] += 1
                    turn_taken[kind].add(key)
                bump(opp_avail, opp_taken, kind, key, per_select[row],
                     bool(was))

    print(f"\ndemonstrator corpus {ds}: {len(paths)} shards, "
          f"{n_selects} selects\n")
    _table(avail, taken, turn_avail, turn_taken, opp_avail, opp_taken)
    return 0


def _table(avail: Counter, taken: Counter, turn_avail: dict,
           turn_taken: dict, opp_avail: dict = None,
           opp_taken: dict = None) -> None:
    """Per-select rate, per-turn rate, and per-opportunity rate.

    Per-select understates: these options stay legal across every select of a
    turn, so declining one to play a supporter first counts as a miss. Per-turn
    fixes that but overstates any repeatable line -- with two Munkidori, one
    Adrena-Brain scores 100%. `opps` is the honest denominator: activations
    offered in the turn (see MULTIPLICITY). `*` marks a "count" line, i.e. one
    where `opps` and `turns` can actually differ."""
    print(f"{'opportunity':<32}{'selects':>9}{'rate':>8}"
          f"{'turns':>9}{'rate':>8}{'opps':>9}{'rate':>8}")
    for kind in sorted(avail, key=lambda k: -avail[k]):
        n, t = avail[kind], taken[kind]
        tn = len(turn_avail.get(kind, ()))
        tt = len(turn_taken.get(kind, ()))
        turn_rate = f"{tt / tn:.1%}" if tn else "-"
        on = ot = 0
        if opp_avail is not None:
            per_turn = opp_avail.get(kind, {})
            uses = opp_taken.get(kind, {}) if opp_taken else {}
            on = sum(per_turn.values())
            # a turn cannot take more than it was offered
            ot = sum(min(uses.get(k, 0), v) for k, v in per_turn.items())
        opp_rate = f"{ot / on:.1%}" if on else "-"
        mark = "*" if multiplicity(kind) == "count" else " "
        print(f"{kind + mark:<32}{n:>9}{t / n:>8.1%}"
              f"{tn:>9}{turn_rate:>8}{on:>9}{opp_rate:>8}")
    if not avail:
        print("  (no tracked opportunity ever appeared)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="audit the training shards instead of "
                                     "playing games (e.g. artifacts/pds_v2)")
    ap.add_argument("--agent", default="bc")
    ap.add_argument("--opponent", default="rule:v10,noS")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--deck-b", default="lucario_v10")
    ap.add_argument("--matches", type=int, default=100)
    args = ap.parse_args()

    if args.corpus:
        return audit_corpus(args.corpus)

    _, deck_a = arena.resolve_deck(args.deck)
    _, deck_b = arena.resolve_deck(args.deck_b)
    name_a, agent_a = arena.build_agent(args.agent, deck_a)
    name_b, agent_b = arena.build_agent(args.opponent, deck_b)

    avail: Counter = Counter()
    taken: Counter = Counter()
    turn_avail: dict = {}
    turn_taken: dict = {}
    game = [0]
    tgt_n: Counter = Counter()
    tgt_low: Counter = Counter()
    opp_avail: dict = {}
    opp_taken: dict = {}
    spread: Counter = Counter()
    audited = wrap(agent_a, avail, taken, turn_avail, turn_taken, game,
                   tgt_n, tgt_low, opp_avail, opp_taken, spread)

    wins = 0
    for m in range(args.matches):
        game[0] = m
        # seat-swap so setup-order effects cancel, as arena.py does
        if m % 2 == 0:
            r = harness.play_game(audited, agent_b, list(deck_a), list(deck_b))
            wins += (r.winner == 0)
        else:
            r = harness.play_game(agent_b, audited, list(deck_b), list(deck_a))
            wins += (r.winner == 1)

    print(f"\n{name_a} [{args.deck}] vs {name_b} [{args.deck_b}], "
          f"{args.matches} games  (won {wins})\n")
    _table(avail, taken, turn_avail, turn_taken, opp_avail, opp_taken)
    if tgt_n:
        print(f"\n{'target choice (>=2 opp Pokemon)':<32}{'selects':>9}"
              f"{'chose lowest HP':>18}")
        for kind in sorted(tgt_n, key=lambda k: -tgt_n[k]):
            n = tgt_n[kind]
            print(f"{kind:<32}{n:>9}{tgt_low[kind] / n:>18.1%}")
        print("  (chance is ~1/k for k candidates; low here means we are not "
              "finishing the cheap target)")
    if spread:
        total = sum(spread.values())
        print(f"\ndark attach with a bare AND a loaded Munkidori offered "
              f"(n={total})")
        for k in sorted(spread, key=lambda k: -spread[k]):
            print(f"  {k:<34}{spread[k]:>7}{spread[k] / total:>8.1%}")
        print("  (a 2nd {D} on a Munkidori buys nothing; the 1st on a bare one "
              "buys a whole Adrena-Brain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
