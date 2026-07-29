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

**P4 is the job.** The user watched the live 936 agent and flagged three things.
All three have the P2c signature — decisions requiring HP/damage arithmetic the
net's features cannot represent — and P2c was worth 184 LB points.

**Do them in this order. Verify mechanics before writing any rule.**

| | item | state | cost |
|---|---|---|---|
| 0 | Second LB reading on the 936; fill in its ref | free | minutes |
| 1 | **P4a** — Boss's Orders → drag a KO-able benched target | not started | hours |
| 2 | **P4b** — spread {D} energy across two Munkidori | not started | hours |
| 3 | **P4c** — fix the audit's per-turn blindness | not started, *unblocks 4a/4b* | ~1 h |
| 4 | **P1** — re-rank decks against `rule:v10` | not started | ~20 min |
| 5 | **P2** — MAIN-decision rules for the chosen deck | not started | days |
| 6 | **P3** — abomasnow / Crustle lockdowns | not started, fold into P2 | hours |

Replays of the live 936 agent vs real opponents are at
**`replays/submission_replay_2026-07-29/`** (user-supplied). These are the only
games we have against the *actual* LB field rather than our six local rule
agents — use them for diagnosis, not training (§6: more imitation data is dead).

### P4a — Boss's Orders should drag something we can kill

**User observation:** *"we had the chance to play Boss's Orders to bring out a
weaker benched Pokémon to the active spot and knock it out with Shadow Bullet
but we didn't."*

**Why this is credible and not already closed.** P2d measured Boss's Orders
*frequency* at 38.2% of legal turns against demonstrators' 31.4% — we play it
**more** than they do. That closed "we lack access." It says nothing about
**which Pokémon we drag**, and target choice is precisely the axis P2c proved
the net is blind on: `optfeat` has no HP, so "this benched Pokémon dies to
Shadow Bullet" is unrepresentable. Same bug, different select.

**Note the compound structure** — this is harder than P2c. P2c overrode a single
select in isolation. This one is *Boss's Orders + attack* as a two-step plan:
the gadget is only worth playing if a specific bench target dies to a specific
attack this turn. `targeting.py` fires on a select it can answer locally; P4a
needs damage arithmetic across two decisions.

**Do:** (1) instrument the Boss's Orders target select the way P2c was
instrumented — what fraction of the time do we drag a target that our available
attack can KO, when such a target exists? (2) `agents/sa/textdmg.py` already
computes expected damage — reuse it, do not rewrite. (3) Rule goes in
`targeting.py` alongside `chip_target`.

### P4b — Two Munkidori want one energy each, not two on one

**User observation:** *"we sometimes put two dark energies on Munkidori when we
had two Munkidoris on the bench. If we had attached one each we'd have been able
to use Adrena-Brain twice."*

**The arithmetic, if the reading is right:** Adrena-Brain (Munkidori, card 112)
is *"Once during your turn, if this Pokémon has any {D} Energy attached, move up
to 3 damage counters from 1 of your Pokémon to 1 of your opponent's."* The
energy condition is **"this Pokémon"** — per copy. So 1 energy on each of two
Munkidori = **two activations = 6 counters = 60 damage relocated**; 2 energy on
one = one activation = 30. The second energy on a single Munkidori is doing
nothing at all. The user is very likely right.

**Verify in-engine before building on it** — three things, all cheap:
1. Is the ability once **per Pokémon** per turn, or once per turn globally? The
   whole idea dies if it is global.
2. Does one {D} suffice, i.e. is it a threshold not a cost? ("has any" reads as
   a threshold — nothing is discarded.)
3. **Is Munkidori a "Marnie's Pokémon"?** This decides whether spreading is
   cheap or slow. Grimmsnarl ex's `Punk Up` searches up to 5 Basic {D} and
   attaches to *your Marnie's Pokémon*. If Munkidori qualifies, Punk Up can
   spread in one shot. If not, energy comes from the 1-per-turn manual attach
   and the spread costs two turns — which makes getting the *first* attachment
   right much more valuable, and makes the rule simple: **never attach a second
   {D} to a Munkidori that already has one while another Munkidori has zero.**

**Do:** verify (1)–(3) against the SDK card text and a live engine probe, then
write the rule as an ATTACH-select override in `targeting.py`.

### P4c — The audit is blind to exactly these bugs (instrument fix)

**User observation:** *"I think we are not using Adrena-Brain at every chance."*

**This directly contradicts a measurement we already trusted, and the user is
right — the instrument is wrong.** `scripts/opportunity_audit.py` reported
`munkidori_adrena_brain` at **99.3%** for our clone vs 97.2% for demonstrators,
which reads as "already maxed, nothing to find here."

That number is computed **per turn, as a binary**: did we use it at all this
turn? **With two Munkidori on the bench, using it once scores 100%** — a missed
second activation is completely invisible to the metric. P4b and P4c are the
same bug seen from two sides, and the audit cannot see either. The same
blindness applies to `dark_energy_to_munkidori` (78.3%): per-turn binary cannot
see *which* Munkidori got the energy.

**Generalisable lesson, add to §2 as rule 8:** a per-turn binary "did we take
this line" metric cannot detect under-use of a repeatable ability or
misallocation among identical copies. **Count opportunities, not turns.**

**Do:** change the audit to count *activations available vs activations taken*
per turn, not `any()`. Then re-run it over
`replays/submission_replay_2026-07-29/` and the demonstrator shards. This
re-opens the P2b table — those "already at demonstrator parity" verdicts are
only trustworthy for once-per-turn lines.

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
- `targeting.py` — **the +184 LB rule.** Overrides DAMAGE / DAMAGE_COUNTER /
  DAMAGE_COUNTER_ANY selects, which the net cannot answer because `optfeat`
  gives it no HP. Fires only when every option is an opponent Pokemon.
  **Every P4 rule belongs here.**
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

# A/B a rule override against the pure clone (how P2c was measured; how to measure P4)
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
