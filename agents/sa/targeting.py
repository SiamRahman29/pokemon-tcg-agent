"""Aim chip damage at the Pokemon it can actually finish.

Measured defect (`scripts/opportunity_audit.py`, 80 games): when the clone
chooses which of the opponent's Pokemon to point an effect at, it picks the
lowest-HP candidate 25.7% of the time for Adrena-Brain's counter move and 42.1%
for Shadow Bullet's bench snipe. With 2-4 candidates on board that is chance.

The cause is in the features, not the weights: `optfeat.option_features` gives
the net a card-id embedding and eight positional scalars per option, and **no
HP and no damage** -- so the net cannot represent "this one dies to 30" at all.
No amount of training fixes that. Retraining with an HP feature would, but it
bumps `optfeat.VERSION` and invalidates every existing net; this is the cheap
half of the fix, and it is deterministic besides.

Both effects deal exactly 30: Shadow Bullet's "30 damage to 1 of your
opponent's Benched Pokemon", and Adrena-Brain's "move up to 3 damage counters".
So the rule is: if anything dies to 30, kill the one worth the most prizes;
otherwise concentrate on the one closest to dying.

Deliberately narrow. It fires only when *every* option in the select resolves
to an opponent Pokemon, which leaves the mixed selects (Adrena-Brain's "from 1
of YOUR Pokemon" source pick) to the net.
"""
from __future__ import annotations

from . import cards

# SelectContext ints (mirror cg.api without importing it)
DAMAGE_COUNTER = 13
DAMAGE_COUNTER_ANY = 14
DAMAGE = 15
CHIP_CONTEXTS = (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE)

# AreaType ints
_ACTIVE, _BENCH = 4, 5

# Shadow Bullet's bench snipe and Adrena-Brain's 3 counters are both 30.
CHIP_DAMAGE = 30


def _pokemon_at(state: dict, player: int, area: int, index: int) -> dict | None:
    try:
        pl = state["players"][player]
        if area == _ACTIVE:
            act = pl["active"]
            return act[0] if act and act[0] is not None else None
        if area == _BENCH:
            bench = pl["bench"]
            if 0 <= index < len(bench):
                return bench[index]
    except (KeyError, IndexError, TypeError):
        return None
    return None


def chip_target(obs: dict) -> list[int] | None:
    """Ranked option indices for an opponent-targeting select, else None."""
    sel = obs.get("select") or {}
    if sel.get("context") not in CHIP_CONTEXTS:
        return None
    options = sel.get("option") or []
    if len(options) < 2:
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None

    scored: list[tuple[tuple, int]] = []
    for i, opt in enumerate(options):
        player = opt.get("playerIndex")
        if player is None or player == me:
            return None  # mixed or own-side select: leave it to the net
        pk = _pokemon_at(state, player, opt.get("area"), opt.get("index") or 0)
        if pk is None:
            return None  # something we cannot read: leave it to the net
        hp = pk.get("hp")
        if hp is None:
            return None
        kills = hp <= CHIP_DAMAGE
        # kills first, most prizes among those, then closest to dying
        scored.append(((0 if kills else 1,
                        -cards.prize_value(pk["id"]) if kills else 0,
                        hp), i))

    scored.sort()
    return [i for _, i in scored]
