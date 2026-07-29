# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** win the public LB. Top is **1179.6** (`flg`). We are at **936**.
Deadline **2026-08-16**, then ~2 weeks of continued play. Kaggle CLI is
authenticated, user has entered.

**Read §2 before trusting any number. §3 is the live plan.**
**This file must always end with a live plan, never a summary.**

---

## 1. Where we are (day 4, 2026-07-29)

**The targeting fix landed and it was worth ~184 LB points.**

| submission | what | LB |
|---|---|---|
| `55054446` | **clone v2 + `targeting.py`** — live | **936.0 / 916.8** ✅ |
| `55048039` | clone v2, no targeting | 752 → 758.6 (settled) |
| `55049206` | `rule:iono` sample agent | ~700–716 (settled) |

✅ **Confirmed under rule 2**: 936.0 and 916.8, readings 8 h apart, both far
above the 758.6 of the same clone without `targeting.py`.

That jump came from **one missing feature**, not from more training. `optfeat`
gives the net no HP and no damage, so it could not represent "this one dies to
30" and aimed chip damage at chance (25.7% lowest-HP picks). `agents/sa/targeting.py`
overrides those selects with a rule. Arena said `bc` vs `bc:noChip` = 0.577
[0.555, 0.599] n=2000 (+54 Elo) and `bc` vs `rule:v10,noS` = 0.418 → **0.537**
[0.506, 0.568] n=1000, flipping us past the public LB-950 agent. The LB agreed.

**This is now the project's central method, confirmed end-to-end:**
> Find decisions the features cannot express, and write a rule for them.
> Three separate axes of *more training* bought nothing. One missing feature
> bought 184 LB points.

§3 is three more candidates of exactly that shape, from the user watching the
936 agent's real games.

### What the top of the board does

Nothing strong here is learned. `notebooks/` has three checked-in reference
agents: `strong-start-baseline-agent-v10-lb-950` (LB 950+, hand-written
deck-specific scoring, ~350 readable lines), `rule-based-not-psychic-alakazam-best-5th`
(**5th place, pure rules, no ML, no search**), and
`a-sample-archaludon-75-wr-vs-my-1300-starmie` (author reports 1300+;
matchup rules with grid-searched thresholds). The competition rewards **deck
expertise + matchup rules + damage arithmetic**.

---

## 2. How not to fool yourself

Every rule below was paid for. Rules 1, 2 and 8 have each invalidated real work.

1. **n=24 is noise.** A BC game costs ~0.17 s — n=1000 is 17 s of CPU. **Never
   accept an n<100 strength claim for anything cheap to measure.** ~2pp effects
   need n≈2000.
2. **One LB score is not a result.** `55049206` read 743.0 → 697.4 → 704.1.
   **Require two readings ≥1 h apart that agree.** Only the latest 2 submissions
   play episodes, so older scores are frozen, not converged.
3. **Validation metrics do not predict playing strength here — five times.**
   Value-net loss, policy top-1 ×3, `--winners-only`. **Judge every net in the
   arena, head-to-head**, never by val accuracy.
4. **Compare nets head-to-head, not through a third opponent.**
   `bc:<tag>,net=<path>` runs two nets in one process.
5. **A cross-deck arena score is mostly a DECK MATCHUP, not agent skill.**
   `rule:lucario` scores 0.781 vs `rule:iono`; the ~104-Elo-stronger
   `rule:v10,noS` scores 0.788 — indistinguishable. The pilot is invisible
   through that anchor. Head-to-head they are 0.646 [0.616, 0.675].
   **Measure skill in near-mirror matchups only.**
6. **CPU contention distorts wall-clock-budgeted agents** (`search:*`,
   `rule:v10` without `noS`). BC and `rule:*,noS` are untimed, so cross-run
   comparison is valid for them.
7. **This machine gives ~1.4 cores of real throughput** (Ryzen 5500U, 15 W).
   Run 2–3 jobs, not 4+.
8. **Frequency is not correctness, and per-turn binary audits hide
   multiplicity.** `munkidori_adrena_brain` read 99.4% per *turn* and 96.9% per
   *opportunity* — because with two Munkidori a turn offers two activations and
   `any()` scores one as 100%. **Count opportunities, not turns**
   (`MULTIPLICITY` in `opportunity_audit.py`).
9. **A metric that never prints is not a metric that passed.** `drag_target`
   was in the audit for days reading zero rows: it was keyed on `TO_ACTIVE`, but
   Boss's Orders drags through **`SWITCH`**. `TO_ACTIVE` is our own post-KO
   promotion, so every option was on our side and the opponent-only filter
   dropped them all, silently. **Check that each row has a non-zero
   denominator before believing the table.**

---

## 3. THE PLAN (day 5)

**P4 was the job and it is mostly done.** All three items came from the user
watching the live 936 agent. Two landed, one is negative and being split.

| | item | state | arena |
|---|---|---|---|
| 0 | Second LB reading + ref | **done** — `55054446`, 936.0 / 916.8 | — |
| 3 | **P4c** — audit counts opportunities, not turns | **done** | — |
| 2 | **P4b** — spread {D} across two Munkidori | **done, huge** | **0.702 [0.687, 0.715]** n=4000 |
| 1 | **P4a** — Boss's Orders (drag target + when to play) | **negative as a pair**; isolating | 0.452 [0.435, 0.470] n=3000 |
| 4 | **P1** — re-rank decks against `rule:v10` | not started | ~20 min |
| 5 | **P2** — MAIN-decision rules for the chosen deck | not started | days |
| 6 | **P3** — abomasnow / Crustle lockdowns | not started, fold into P2 | hours |

**Next action: ship P4b.** It is the biggest measured effect this project has
produced — bigger than the +184-point targeting fix — and it is independent of
the P4a question. Do not wait for P4a to resolve before submitting.

Then, in order: (a) finish the P4a split below, (b) validate the shipped config
head-to-head against `rule:v10,noS` at n≥500, (c) P1, (d) P2.

Replays of the live 936 agent vs real opponents are at
**`replays/submission_replay_2026-07-29/`** (user-supplied). These are the only
games we have against the *actual* LB field rather than our six local rule
agents — use them for diagnosis, not training (§6: more imitation data is dead).

### P4b — Spread {D} across two Munkidori — DONE, +148 Elo

**The user's reading was right, and it is the biggest effect measured here yet.**
`targeting.energy_spread`, `bc:noSpread` turns it off.

**arena: `bc` vs `bc:noSpread` = 0.702 [0.687, 0.715], n=4000, grimmsnarl
mirror.** For comparison the +184-LB-point chip-targeting fix scored 0.577.

Four facts, all verified in-engine (`probe_adrena.py` pattern, 40 games with a
wrapper that greedily takes every Munkidori ability):

1. Adrena-Brain is **once per Pokemon**, not once per turn. We activated it
   twice in a turn 35 times; a slot that had used it was never re-offered.
2. The {D} condition is a **threshold, not a cost** — energy after use was
   unchanged 138 times out of 138.
3. **Munkidori is not a "Marnie's Pokemon"** (card 112 is plain `Munkidori`;
   the others are `Marnie's Impidimp/Morgrem/Grimmsnarl ex`). Punk Up cannot
   attach to it: in 40 games every attach option targeting a Munkidori came
   from the hand, i.e. the 1-per-turn manual attach. That is what makes the
   wasted attach expensive.
4. **A second {D} on a Munkidori is dead, full stop.** Munkidori's only attack
   is Mind Bend, cost {P}{C}, and this deck runs zero Psychic energy — so it
   cannot even be attack setup.

So: two Munkidori at 1 {D} each move 6 damage counters a turn (a 60-point swing,
since Adrena-Brain both heals us and damages them); one Munkidori at 2 {D}
moves 3. The clone chose the wasted attach **143 times to 94** — worse than a
coin flip, because `optfeat` gives it no attached-energy count. The rule takes
that to 0 and lifts Adrena-Brain activations from 1.26 to 1.60 per turn.

### P4a — Boss's Orders: NEGATIVE as a pair, being split

**User observation:** *"we had the chance to play Boss's Orders to bring out a
weaker benched Pokémon to the active spot and knock it out with Shadow Bullet
but we didn't."* The observation was accurate; the fix was not.

**arena: `bc` vs `bc:noDrag,noBoss` = 0.452 [0.435, 0.470], n=3000 — the two
rules together cost ~33 Elo.** Both isolations are running
(`out/arena/ab_drag_only.jsonl`, `out/arena/ab_boss_only.jsonl`, n=2000 each).

Two separate rules were written, both in `targeting.py`:

- `drag_target` — rank the drag by (dies to our attack, prizes, lowest HP).
  Small lever: the clone already took the best available KO 85 times out of 99.
- `boss_converts` — **play** Boss's Orders when our attack would not KO the
  opponent's Active but would KO something on their bench. Big lever on paper:
  157 such turns in 300 games, and the clone played it on 36.9% of them (vs
  25.7% of all other legal turns — so it does discriminate, barely). The rule
  takes that to 100%.

**The prior is that `boss_converts` is the guilty one**, and if so it is a
lesson worth writing down: it spends the turn's **Supporter** — the slot that
otherwise plays Petrel or Lillie's Determination, i.e. the deck's whole engine —
to buy one guaranteed prize, often only 1 prize off a small bench sitter. A
correct *local* arithmetic can still be a bad trade globally. Note also that it
fires at the first MAIN select of the turn, before any Rare Candy line could
upgrade our Active into something that KOs the opposing Active outright.

**Both now default to False** in `PolicyAgent` and in `bc:`, and are opt-in via
`bc:drag` / `bc:boss`. The submission's `main.py` does a bare `_A(_deck)`, so a
plain `bc` in the arena is exactly what ships — turn a flag's default on only
after that rule clears 0.5 by itself. The isolation runs were launched under
the older opt-out defaults, so their repro commands today are
`bc:drag` vs `bc:base` and `bc:boss` vs `bc:base`.

### P4c — Count opportunities, not turns — DONE

**User observation:** *"I think we are not using Adrena-Brain at every chance."*
The instrument was wrong, as suspected — but the corrected number is small.

`opportunity_audit.py` now declares a `MULTIPLICITY` per line and prints an
`opps` column beside `turns`. `munkidori_adrena_brain` reads **99.4% per turn
but 96.9% per opportunity** (452 opportunities over 359 turns, 150 games). Real,
but a ~3% miss — the activation itself was never the lever. The lever was
upstream, in P4b: getting a second Munkidori armed at all.

Only `munkidori_adrena_brain` is a `"count"` line today, because it is the only
one whose copies are countable **on both sides** (one ABILITY option per
Munkidori, live and in the shards). Items are repeatable too, but a Rare Candy
option carries no target, so counting options would invent a denominator. The
docstring explains this; do not widen `"count"` without a real target count.

The audit also gained a live-only allocation metric (bare vs loaded Munkidori)
and its `drag_target` row now works at all — see §2 rule 9.

**Still open from this item:** the P2b parity verdicts were only re-derived for
`munkidori_adrena_brain`. The others are once-per-turn lines and so unaffected,
but the demonstrator-corpus side of the new `opps` column has not been run
(`--corpus artifacts/pds_v2`); `artifacts/` is gitignored and may need the
rebuild in §5.

### P1 — Re-rank decks against `rule:v10` (cheap, still unstarted)

The old sweep ranked decks by how well our clone beat `rule:iono`, which rule 5
kills. `mega_lucario_ex` came **last** there yet is the deck the LB-950 agent
plays. `scripts/deck_sweep.ps1` now defaults to `rule:v10,noS` / `lucario_v10`
and covers all 7 decks — run with no arguments (§7 PowerShell gotcha).
~20 min. It answers "what should P2 be written for", **not** "which deck is
strongest".

### P2 — Write a real agent for one deck

Where the leaderboard is. Explicit attack planning (attacker/target/attack index
with weakness and prize arithmetic), per-option scoring, matchup branches. No
search (§6). **Bar: 0.5 against `rule:v10,noS` in a near-mirror, n≥500.**
`rule:v10` is a working readable template and is already in-process. The clone
stays as fallback for every select the rules do not cover — a hybrid starts
strictly above whichever component is better per decision class.

`scripts/context_accuracy.py` says **MAIN holds 3,930 of the net's 6,424 misses**
(18,924 rows, 33.9% miss). The small contexts are picked over; MAIN is the
remaining mass and it is what P2 is for.

### P3 — The abomasnow hole (open)

0.360 vs 0.475–0.519 elsewhere (pre-P2c, re-measure), and our selects/turn
collapse from 12.5–16.6 to **8.6** with shorter games — a lockdown, not subtle
misplay. Replay a loss with `SA_DEBUG=1` and read the actual select options.

**Related and untested: Crustle.** `Mysterious Rock Inn` (card 345) prevents all
damage from opponent {ex} attacks, and Grimmsnarl ex is `ex=True`, so **it deals
literally zero to Crustle** — attacking into it wastes every turn. **There is no
Crustle deck in the repo**, so `attack_into_ex_immune_active` (already in the
audit) has never fired. The out: Adrena-Brain and Freezing Shroud *move/place
damage counters*, which is not "damage done by attacks", so they should bypass
the prevention — **verify in-engine.** `dashimaki360/beating-the-day-1-1-crustle-bot`
is a public notebook on this matchup; V10 hardcodes 344/345 as "the crustle wall".

---

## 4. What ships

`agents/sa/bcagent.py` `PolicyAgent` + `agents/sa/policy_net.npz`
(= `policy_lw2`, listwise, 2,810-game corpus, val top-1 0.6755) +
`agents/sa/targeting.py`. ~1 ms/move, uses 0.1 s of the 600 s pool.

### Code map (`agents/sa/`)

- `bcagent.py` — **what we ship.** `net_path` pins an npz; `chip_targeting`
  toggles the override (`bc:noChip` in the arena). Default True.
- `targeting.py` — **all the rule overrides.** Four of them now, each with its
  own `PolicyAgent` flag and its own `bc:` arena switch, so any one can be A/B'd
  alone. **Every new rule belongs here.**

  | function | select | switch | arena |
  |---|---|---|---|
  | `chip_target` | DAMAGE / DAMAGE_COUNTER(_ANY) | `noChip` | 0.577, n=2000 → **+184 LB** |
  | `energy_spread` | MAIN, {D} ATTACH onto a Munkidori | `noSpread` | **0.702, n=4000** |
  | `drag_target` | SWITCH (Boss's Orders' drag) | `noDrag` | isolating |
  | `boss_converts` | MAIN, plays Boss's Orders | `noBoss` | isolating; pair was **0.452** |

  `chip_target` / `drag_target` replace the whole ranking and fire only when
  *every* option is an opponent's Pokemon. `energy_spread` is different in kind:
  it takes the net's pick and only redirects it, never creating or suppressing
  an attach. `boss_converts` is the only one that forces an action outright,
  which is very likely why the pair measured negative.
- `policynet.py` — numpy inference. `SA_PNET_PATH` env override; **dim guard**
  (stale net → `None` → fallback; never remove it).
- `features.py` (v2, DENSE_DIM=242, PER_SLOT=18) / `optfeat.py` — shared by
  trainer and inference. **Any npz trained pre-v2 fails the dim guard.**
  Adding an HP/damage feature here bumps `VERSION` and retrains every net —
  a serious candidate, but `targeting.py` has been the cheaper path so far.
- `evalfn.py` + `textdmg.py` — handcrafted eval / expected damage. **P4a needs
  `textdmg`.** Same object as V10's `evaluate_state`; read both together.
- `agent.py` (`SearchAgent`), `planner.py`, `timemgr.py` — search path, §6.
- `worlds.py`, `tracker.py`, `fastsearch.py`, `deck_library.json`.
- Both agents never raise: fallback = `list(range(minCount))`.

### The arena's real opponent: `rule:v10`

`scripts/import_v10_agent.py` lifts the LB-950 notebook into
`agents/agentkit/rulebased/sources/v10.py` plus `decks/lucario_v10.py` (its own
retuned 60 — *not* `decks/mega_lucario_ex.py`). Idempotent. Flags: `noS`
disables its MCTS, `tb<sec>` sets its budget — **in practice both are no-ops
because the MCTS never runs (§6)**; pass `noS` anyway so the archived name
records intent. `rule:v10x` makes the search reachable (still falls back).

---

## 5. Commands

```powershell
# LB / submission status
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(s.ref, s.date, s.status, s.public_score, '|', str(s.description)[:60]) for s in a.competition_submissions('pokemon-tcg-ai-battle')[:5]]"

# Skill measurement: near-mirror head-to-head (rule 5). The only kind that counts.
python -X utf8 scripts/arena.py play "rule:v10,noS" rule:lucario `
    --deck-a lucario_v10 --deck-b mega_lucario_ex --matches 500

# Against the real bar
python -X utf8 scripts/arena.py play bc "rule:v10,noS" `
    --deck-a grimmsnarl --deck-b lucario_v10 --matches 500

# A/B a rule override against the pure clone (how every targeting.py rule is judged).
# Switches: noChip, noSpread, noDrag, noBoss -- combine to isolate one rule.
python -X utf8 scripts/arena.py play bc "bc:old,noChip" `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000

# Net A/B, two nets in one process (~5 min, n=2000)
python -X utf8 scripts/arena.py play "bc:new,net=out/policy_X.npz" bc `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 --archive out/arena/ab_X.jsonl

powershell -File scripts/deck_sweep.ps1        # P1; no args (see gotcha)
python -X utf8 scripts/tally.py "<agent>" "out/arena/foo_*.jsonl"

# Audits — run these BEFORE writing any rule
python -X utf8 scripts/opportunity_audit.py --matches 100        # our games
python -X utf8 scripts/opportunity_audit.py --corpus artifacts/pds_v2   # demonstrators
python -X utf8 scripts/context_accuracy.py                       # per-context top-1

# Train (12 epochs; artifacts/pds_v2 is the shipped corpus)
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v2 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 --out out/policy_X.npz

# Build + submit (smoke-tests the bundle the way Kaggle loads it)
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc --nets policy
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_submit('dist/submission.tar.gz','msg','pokemon-tcg-ai-battle')"

# Import public notebook agents
python -X utf8 scripts/import_v10_agent.py     # rule:v10 + decks/lucario_v10
python -X utf8 scripts/import_rule_agents.py   # the four sample agents

# Find new public notebooks (this is how V10 was found — redo periodically)
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(k.ref,'|',k.title) for k in a.kernels_list(competition='pokemon-tcg-ai-battle',sort_by='voteCount',page_size=30)]"

# Rebuild data only (more data is NOT a lever — §6)
python -X utf8 scripts/fetch_top_episodes.py --date 2026-07-26 --max 400
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds/d26 replays/2026-07-26
```

### Data on disk

`replays/`: 07-17..07-22, 07-24, 07-26, 07-27 = 400 each; 07-23 = 175,
07-25 = 268, 07-16 = 115; 07-13/14/15 = 0. Plus 366 old-repo replays at
`E:\Kaggle\pokemon-tcg-simulation\replay_miner\replays\2026-07-06..12`.
**`replays/submission_replay_2026-07-29/` = our live 936 agent vs the real
field** (user-supplied; the only non-local opposition we have).

`artifacts/**` is gitignored. `artifacts/pds/` = 4,010 games (the *rejected*
lw3 corpus); `artifacts/pds_v2/` = 2,810 (the shipped v2 corpus) and exists
only on this disk. `pds_v2` is `pds` minus the three days that made lw3 worse:

```powershell
foreach ($d in @('old','d21','d22','d23','d24','d25','d26','d27')) {
  New-Item -ItemType Directory -Force "artifacts/pds_v2/$d" | Out-Null
  Copy-Item "artifacts/pds/$d/shard_*.npz" "artifacts/pds_v2/$d/" -Force
}
```

If `pds` itself is lost, rebuild shards from `replays/` with
`build_policy_dataset.py`.

---

## 6. Settled — do not redo

**The clone is plateaued. Three training axes are negative** (head-to-head,
n=2000, vs the previously shipped net):

| net | corpus | val top-1 | vs prev shipped |
|---|---|---|---|
| v1 (BCE) | 2,410 | 0.6596 | — |
| **v2 `policy_lw2`** | **2,810** | **0.6755** | **0.524 — SHIPPED** |
| `policy_lw3` (more data) | 4,010 | 0.6933 | 0.491 |
| `policy_win` (`--winners-only`) | 2,810 | 0.6410 | **0.375 — decisive** |

More data, more val accuracy, and winners-only all fail. `--winners-only` is
12pp *worse* — cloning the losing side is **helping**. Note lw3 has the best val
accuracy and lost: rule 3 in action. `--loss listwise` beats pointwise BCE and
reaches in 1 epoch what BCE took 4 to reach.

**Search is out, ours and the field's.** Ours: `search:M,noV,roll,mo,mc20,pb0.15`
vs `bc` = 0.323, n=31 — a terminal rollout returns 0/1, so a mean over 12
determinizations has SE ≈ 0.14 and the max over ~9 rivals sits ~0.21–0.28 above
truth by chance; it overruled the clone on 52% of decisions. More
determinizations and the value net were also negative. **V10's shallow MCTS has
never once executed** — two independent bugs (its candidate set comes from
`choose()` truncated to `select.maxCount`, which is 1 for every MAIN select
measured 70/70; and `search_begin(obs, your_deck=yd)` passes 1 of 7 required args
and raises `TypeError` into a bare `except`). Confirmed by timing: 200 games in
11.8 s. **So LB 950+ is 100% handcrafted policy, and nothing in this competition
has ever demonstrated search is worth anything.** Loose end if ever revisited:
`agents/sa/worlds.py`'s `World` is exactly the `search_begin` argument bundle.

**Ruled out by measurement:**

- **Decklist changes** — +2 Boss's Orders / -1 Tool Scrapper / -1 Spikemuth Gym
  scored 0.490 [0.468, 0.512], n=2000. We already play Boss's Orders on 38.2% of
  legal turns vs demonstrators' 31.4%; **Team Rocket's Petrel (4x) tutors any
  Trainer**, so access is already there. Spikemuth Gym is played ~100% by both
  sides — do not cut it. The list is an exact 60 seen 290× in one day's top
  episodes and the net is trained on it, so variants are off-distribution too.
- **TO_HAND duplicate-avoidance** — demonstrators fetch a duplicate 5.8%
  (n=57,053), we fetch one 5.8% (n=482). Already correct.
- **REMOVE_DAMAGE_COUNTER** — lowest lift on the board, but demonstrators are
  themselves inconsistent (Active 33.6%, max-prize 60.6%, ~2.8 options,
  n=9,911). **A low lift can mean a noisy label, not a blind feature.**
- **Self-play RL — dropped.** Days of work on 1.4 cores to maybe reach where
  hand-written rules already sit. Nothing at the top of this board is learned.

**Do not resurrect:** the arena→LB ladder anchored on `rule:iono`; the old deck
sweep's ranking; "the clone is comfortably above the rule baseline"; every n=24
number and every strength claim dated before 2026-07-27 pm (measured through
stale nets silently rejected by dim guards, a compute knob that could not bind,
and a mirror matchup compared against cross-deck runs); "3× compute made it
worse" (`SA_SPEND_MULT` only grants time, and time was never binding).

⚠ **Per §3c, the P2b "already at demonstrator parity" verdicts
(`munkidori_adrena_brain` 99.3%, `rare_candy_play` 82.0%,
`evolve_impidimp_to_morgrem` 91.6%, `dark_energy_to_munkidori` 78.3%) are only
valid for once-per-turn lines.** Re-derive them after the P4c instrument fix.

---

## 7. Gotchas (all paid for)

- **`__file__` DOES NOT EXIST on Kaggle.** `kaggle_environments/agent.py` does
  `exec(code_object, env)` → `NameError` → ERROR before the agent runs. This
  killed `55028078`. The smoke test `exec`s the source with no `__file__` in
  globals, exactly as Kaggle does — **keep it that way**.
- **Kaggle sets no env vars.** `SA_NO_PNET`/`SA_NO_VNET`/`SA_PNET_PATH` are inert
  there, so any bundled `.npz` is LIVE. Pin with `--nets none|policy|value|both`.
- **Do not set `SA_COUNT_MODE=expect` with a listwise net** — it assumes
  calibrated probabilities; listwise gives a valid *ranking* only.
- **Kaggle enforces the 600 s pool** (exhausted = loss) though the harness does
  not. `arena.py` records `pool0`/`pool1` and warns below 300 s. BC uses 0.1 s.
  ⚠ **If the machine sleeps mid-run, one game eats the whole nap** and
  `arena.py` prints `WOULD TIME OUT ON KAGGLE` off that single game. Check the
  distribution before believing it: in `ab_spread.jsonl` the worst pool was
  −3606.9 s and the *next* worst was 599.2 s, median 599.9 s, p99 latency 1.6 ms.
- **Submission:** `.tar.gz`, `main.py` + `deck.csv` at TOP level (+ `cg/`, `sa/`).
  Cap 197.7 MiB. 5/day, **latest 2 active**. New submissions start μ=600.
  Validation episode is self-play first — a crash there means Error.
  `kaggle competitions submit` may 400 despite a 100% upload; the Python client
  works, and that call **submits** — it is not a dry run.
- Kaggle Python API returns **snake_case** (`public_score`, `team_name`);
  `competition_leaderboard_view` paginates at 20 rows.
- **PowerShell `-File script.ps1 -Days a,b,c` does not bind an array.** Edit the
  script default and launch with no args. **Never name a param `$Matches`** —
  collides with the automatic regex variable.
- Windows: `python -X utf8` everywhere. Run from repo root; `sys.path` needs
  `src/`, `agents/`, root. Launch long jobs with `Start-Process` (detached) and
  pass `-u` or python block-buffers redirected stdout.
- Some replays download truncated (exactly 3 MiB) and fail JSON parse; builders
  skip them (`errors=N`). Delete + re-fetch to recover.
- Old repo `E:\Kaggle\pokemon-tcg-simulation` = failed pure-RL attempts. Take its
  replays, not its approach.
- **Submission discipline:** submit only what has won head-to-head at n≥500
  against `rule:v10,noS`. Always `--nets`-pin the config. Rebuild rather than
  trusting an old tarball in `dist/`.
- Commit style: fine-grained, one-line semantic messages + Claude co-author
  trailer.
