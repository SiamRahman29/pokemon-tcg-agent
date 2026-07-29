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

`energy_spread` is the same idea one select over: the net cannot see how much
energy an ATTACH target already carries, so it stacks a second {D} on a
Munkidori that already has one instead of arming a bare one.
"""
from __future__ import annotations

from . import cards
from .textdmg import estimate

# SelectContext ints (mirror cg.api without importing it)
DAMAGE_COUNTER = 13
DAMAGE_COUNTER_ANY = 14
DAMAGE = 15
CHIP_CONTEXTS = (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE)

# AreaType ints
_HAND, _ACTIVE, _BENCH = 2, 4, 5

MAIN = 0            # SelectContext.MAIN
SWITCH = 3          # SelectContext.SWITCH -- what Boss's Orders drags with
OPT_PLAY = 7        # OptionType.PLAY
OPT_ATTACH = 8      # OptionType.ATTACH
BOSS_ORDERS = 1182
MUNKIDORI = 112
DARK_ENERGY = 7     # Basic {D} Energy, card id
DARK_TYPE = 7       # ... and energy type; pk["energies"] holds types

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


def _hand_card_id(state: dict, me: int, index: int) -> int:
    try:
        hand = state["players"][me]["hand"]
        card = hand[index]
        return card["id"] if card else 0
    except (KeyError, IndexError, TypeError):
        return 0


def _dark(pk: dict) -> int:
    return sum(1 for e in (pk.get("energies") or []) if e == DARK_TYPE)


def energy_spread(obs: dict, chosen: list[int]) -> list[int] | None:
    """Redirect a wasted second {D} on a Munkidori to a bare one.

    Verified in-engine over 40 games: Adrena-Brain is once **per Pokemon**
    (we activated it twice in a turn 35 times, and a slot that had used it was
    never re-offered), and its "has any {D} Energy attached" is a **threshold,
    not a cost** -- the energy is never consumed (n=138, unchanged every time).
    So two Munkidori holding one {D} each move 6 damage counters a turn; one
    Munkidori holding two moves 3, and the second {D} does nothing else either:
    Munkidori's only attack is Mind Bend, cost {P}{C}, and this deck runs zero
    Psychic energy. Nor is Munkidori a *Marnie's* Pokemon, so Grimmsnarl ex's
    Punk Up cannot attach to it -- the 1-per-turn hand attach is the only
    source, which is what makes spending it on a no-op expensive.

    Measured on the shipped clone (`opportunity_audit.py`, 150 games): when a
    select offered both a bare and an already-loaded Munkidori and the clone
    attached to one of them, it picked the loaded one **143 times to 94** --
    worse than a coin flip, because `optfeat` gives it no attached-energy count.

    Narrow on purpose: this only reorders *which Munkidori* gets the energy the
    net already decided to attach. It never creates or suppresses an attach.
    """
    sel = obs.get("select") or {}
    if sel.get("context") != MAIN:
        return None
    options = sel.get("option") or []
    if not chosen:
        return None
    pick = chosen[0]
    if not 0 <= pick < len(options):
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None

    bare: list[int] = []
    loaded: list[int] = []
    for i, opt in enumerate(options):
        if opt.get("type") != OPT_ATTACH:
            continue
        if (opt.get("area") or _HAND) != _HAND:
            continue
        if _hand_card_id(state, me, opt.get("index") or 0) != DARK_ENERGY:
            continue
        pk = _pokemon_at(state, me, opt.get("inPlayArea"),
                         opt.get("inPlayIndex") or 0)
        if pk is None or pk.get("id") != MUNKIDORI:
            continue
        (bare if _dark(pk) == 0 else loaded).append(i)

    if pick not in loaded or not bare:
        return None
    return [bare[0]] + [i for i in chosen[1:] if i != bare[0]]


# --- Boss's Orders: drag something we can actually kill -------------------

def best_damage(active: dict, mypl: dict, oppl: dict, target: dict) -> float:
    """Best damage `active` can pay for right now, applied to `target`.

    `estimate` is text-pattern-based and approximate in general, but every
    attack this deck can pay for is flat damage (Shadow Bullet 180, Corkscrew
    Punch 60, Frost Smash 60), so here it is exact. Where it is not, it
    under-reads, which only makes the callers below more conservative."""
    if not active:
        return 0.0
    atk_type = cards.card(active["id"]).get("energyType")
    weak = cards.card(target["id"]).get("weakness")
    mult = 2.0 if (weak is not None and weak == atk_type) else 1.0
    best = 0.0
    for aid in cards.card(active["id"]).get("attacks") or []:
        a = cards.attacks().get(aid)
        if a and cards.energy_satisfied(a["energies"], active["energies"]):
            best = max(best, estimate(aid, active, mypl, oppl) * mult)
    return best


def drag_target(obs: dict, prefer_high_hp: bool = False) -> list[int] | None:
    """Ranked option indices for Boss's Orders' drag, else None.

    Boss's Orders resolves through a **SWITCH** select (not TO_ACTIVE, which is
    our own post-KO promotion) whose options are the opponent's benched
    Pokemon. Same blind spot as `chip_target`: no HP in the features, so the
    net cannot tell which of them dies to the attack we are about to make.

    Measured on the shipped clone (300 games): given a KO-able bench target it
    took the best available KO 85 times out of 99 -- 12 drags of a Pokemon that
    survives, 2 that took fewer prizes than were on offer.

    Rank: dies to our attack first, most prizes among those, then closest to
    dying. Same guard as `chip_target` -- every option must be an opponent's
    benched Pokemon, so our own retreats and switches are left to the net.

    `prefer_high_hp` (`bc:drag,dragHi`) flips the tiebreak **inside the KO-able
    group only**: they all die this turn, so "closest to dying" buys nothing
    there, and the user's argument is that the big one is a developing threat
    while a small basic is cheap for them to replace. The non-KO-able fallback
    keeps ascending HP -- there "closest to dying" is the whole point, and
    flipping it would be a different intervention."""
    sel = obs.get("select") or {}
    if sel.get("context") != SWITCH:
        return None
    options = sel.get("option") or []
    if len(options) < 2:
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None
    try:
        mypl, oppl = state["players"][me], state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return None
    active = mypl["active"][0] if mypl.get("active") else None

    scored: list[tuple[tuple, int]] = []
    for i, opt in enumerate(options):
        if opt.get("playerIndex") in (None, me) or opt.get("area") != _BENCH:
            return None
        pk = _pokemon_at(state, 1 - me, _BENCH, opt.get("index") or 0)
        if pk is None or pk.get("hp") is None:
            return None
        kills = best_damage(active, mypl, oppl, pk) >= pk["hp"]
        hp = -pk["hp"] if (kills and prefer_high_hp) else pk["hp"]
        scored.append(((0 if kills else 1,
                        -cards.prize_value(pk["id"]) if kills else 0,
                        hp), i))

    scored.sort()
    return [i for _, i in scored]


def full_rank(net, obs: dict) -> list[int]:
    """The net's complete ranking, not the top-k that `choose` returns.

    A veto needs the runner-up, and every MAIN select measured here has
    maxCount == 1. Plain sort rather than argsort so this module stays
    numpy-free."""
    scores = net.scores(obs)
    return sorted(range(len(scores)), key=lambda i: -float(scores[i]))


def boss_veto(obs: dict, chosen: list[int], rank) -> list[int] | None:
    """Suppress Boss's Orders when their bench holds nothing we can KO.

    **This is the third Boss's Orders intervention, and the only untested one.**
    P4a measured *forcing* the play when it converts (0.493) and *aiming* the
    drag (0.489), both null. Neither touched the case here: the play happening
    at all when it buys nothing. Measured with `scripts/p5_audit.py --matches
    200`: **35 of 108 Boss's Orders plays (32.4%) had no KO-able target on the
    opponent's bench at all.** Those hand the opponent a free promotion --- the
    user watched one drag get evolved into their main attacker the next turn.

    Shape: this deletes an option rather than picking a side in a trade, but the
    option is only *conditionally* dominated (a drag can still strand their
    attacker in the Active, which `best_damage` cannot see), so it sits between
    the P4b class and the P4a class. Per rule 10 it lives or dies by its A/B.

    `rank` is a zero-argument callable returning the net's FULL ranking. It is
    needed because MAIN selects have maxCount == 1 --- `choose` hands back one
    index, so vetoing it leaves nothing to play instead. Called only when the
    veto actually fires, which is why it is lazy.
    """
    sel = obs.get("select") or {}
    if sel.get("context") != MAIN:
        return None
    options = sel.get("option") or []
    if not chosen:
        return None
    pick = chosen[0]
    if not 0 <= pick < len(options):
        return None
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None
    opt = options[pick]
    if opt.get("type") != OPT_PLAY:
        return None
    if _hand_card_id(state, me, opt.get("index") or 0) != BOSS_ORDERS:
        return None
    try:
        mypl, oppl = state["players"][me], state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return None
    active = mypl["active"][0] if mypl.get("active") else None
    if not active:
        return None
    for pk in oppl.get("bench") or []:
        if pk and pk.get("hp") is not None \
                and best_damage(active, mypl, oppl, pk) >= pk["hp"]:
            return None  # the drag buys a prize: let it through

    # Nothing on their bench dies. Fall through to the net's next choice,
    # skipping every other Boss's Orders copy in hand for the same reason.
    vetoed = {i for i, o in enumerate(options)
              if o.get("type") == OPT_PLAY
              and _hand_card_id(state, me, o.get("index") or 0) == BOSS_ORDERS}
    rest = [i for i in rank() if i not in vetoed]
    return rest[:len(chosen)] or None


def boss_converts(obs: dict) -> list[int] | None:
    """Play Boss's Orders when the drag turns a nothing turn into a prize.

    The frequency question was already closed -- we play Boss's Orders on 32.4%
    of legal turns against the demonstrators' 31.4%. The open one was *when*.
    Over 300 games there were 157 turns where our attack would not KO the
    opponent's Active but would KO something on their bench, and the clone
    played Boss's Orders on only 58 of them (36.9%; it plays it on 25.7% of all
    other legal turns, so it does discriminate -- just barely).

    Fires only on that exact shape: we can attack, the Active survives, a
    benched Pokemon does not. It costs the turn's Supporter, which is why it
    stays pinned to the case where the payoff is a guaranteed prize."""
    sel = obs.get("select") or {}
    if sel.get("context") != MAIN:
        return None
    options = sel.get("option") or []
    state = obs.get("current") or {}
    me = state.get("yourIndex")
    if me is None:
        return None
    boss = [i for i, o in enumerate(options)
            if o.get("type") == OPT_PLAY
            and _hand_card_id(state, me, o.get("index") or 0) == BOSS_ORDERS]
    if not boss:
        return None
    try:
        mypl, oppl = state["players"][me], state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return None
    active = mypl["active"][0] if mypl.get("active") else None
    opp_active = oppl["active"][0] if oppl.get("active") else None
    if not active or not opp_active:
        return None
    if best_damage(active, mypl, oppl, opp_active) >= opp_active["hp"]:
        return None  # the Active already dies; the drag would trade down
    for pk in oppl.get("bench") or []:
        if pk and best_damage(active, mypl, oppl, pk) >= pk["hp"]:
            return [boss[0]]
    return None
