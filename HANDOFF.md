# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** win the public LB. Top is now **1179.6** (`flg`); we are at ~750.
Deadline **2026-08-16**, then ~2 weeks of continued play. Kaggle CLI is
authenticated and the user has entered. **This file must always end with a live
plan, never a summary.**

---

## 1. Where we are (day 3, 2026-07-28 pm)

Day 3 read the live baseline (P0), then went and looked at what the *field* is
actually doing. Both answers were bad news, and together they redirect the
project.

**P0 answer: the clone is at parity with the rule baseline, not above it.**
Two readings ≥1 h apart:

| ref | what | 16:09 | 16:54 |
|---|---|---|---|
| `55049206` | **`rule:iono` LIVE baseline** | 743.0 | 697.4 |
| `55048039` | **clone v2 — shipped** | 745.7 | 751.9 |

`55049206` is still moving, so treat it as "somewhere in 700–743". But the
qualitative answer is stable and it is the *opposite* of day 2's hypothesis:
the old `rule:iono` = 763.7 was **not** an inflated stale number, and two days
of behavioural cloning have produced an agent level with a sample rule agent.

**The field answer: a public notebook beats us, and it is not close.**
`romanrozen/strong-start-baseline-agent-v10-lb-950` is a copy-pasteable public
agent measured at **LB 950+**. It is now imported as `rule:v10` (§3).

> **`bc` vs `rule:v10,noS`: 0.418 [0.388, 0.449], n=1000.**
> We lose to a public baseline — with its search *switched off*.

So the ~430-point gap to #1 is not a compute gap and not an "RL would fix it"
gap. We are below freely available work.

### What the top of the board is actually doing

Read `notebooks/` — three reference agents are now checked in:

- `strong-start-baseline-agent-v10-lb-950` (LB 950+) — a large hand-written
  deck-specific scoring policy, **plus a shallow MCTS whose leaf evaluator is a
  dense handcrafted `evaluate_state`** (prize diff ×10000, board strength,
  predicted opponent damage next turn, lethal-threat penalty, anti-stall deck
  conservation). It rolls out **only the rest of the current turn**, never to
  terminal.
- `rule-based-not-psychic-alakazam-best-5th` — **5th place, pure rules, no
  search, no ML.** Its markdown is a page of explicit expert play principles
  (exact Powerful Hand hand-size arithmetic, per-opponent tech cards).
- `a-sample-archaludon-75-wr-vs-my-1300-starmie` — author reports a **1300+**
  Starmie/Froslass submission; the notebook is matchup-specific rules with
  **grid-searched thresholds** ("Lucario threshold = 270, grid searched N=500").

None of the strong agents are learned. The competition is rewarding **deck
expertise + matchup-specific rules + damage arithmetic**. That is the lever we
have not pulled at all.

---

## 2. How not to fool yourself (read this before trusting any number)

Every rule below was paid for. **Rule 8 is new and it invalidates a lot.**

1. **n=24 is noise.** A BC game costs ~0.17 s, so n=1000 is 17 seconds of CPU.
   **Never accept an n<100 strength claim for anything cheap to measure.** For a
   ~2pp effect you need n≈2000.
2. **A fresh LB score is not a result.** `55049206` read 743.0 then 697.4 an
   hour later. **Require two readings ≥1 h apart that agree.** And *frozen ≠
   converged*: only the latest 2 submissions play episodes.
3. **Old submission scores are not comparable to new ones** — except that this
   time the old one was right. Re-measure rather than assume either way.
4. **Validation metrics do not predict playing strength here — five times now.**
   Value-net val loss, policy top-1 three times, and now `--winners-only` (§4).
   Judge every net in the arena, head-to-head, before shipping.
5. **Compare nets head-to-head, not through a third opponent.** `bc:<tag>,net=
   <path>` plays two nets in one process.
6. **CPU contention distorts wall-clock-budgeted agents** — `search:*` and
   `rule:v10` without `noS`. Use same-process head-to-head for those. BC and
   `rule:*,noS` agents are not timed, so cross-run comparisons are valid.
7. **This machine delivers ~1.4 cores of real throughput** (Ryzen 5500U, 15 W).
   Run 2–3 jobs, not 4+. Prefer BC experiments (0.3 s/game) over search ones.
8. **A cross-deck arena score is mostly a DECK MATCHUP, not agent skill.**
   Measured today:

   | matchup | score | what it is |
   |---|---|---|
   | `rule:lucario` vs `rule:iono` | 0.781 [0.755, 0.806] | sample agent, sample deck |
   | `rule:v10,noS` vs `rule:iono` | 0.788 [0.762, 0.812] | LB-950 agent, tuned deck |
   | `rule:v10,noS` vs `rule:lucario` | **0.646 [0.616, 0.675]** | near-mirror = real skill |

   V10 and the sample agent score *identically* against `rule:iono` (0.788 vs
   0.781) despite being ~104 Elo apart. The 0.78 is the lucario deck beating the
   iono deck; the pilot is invisible. **Anchoring on `rule:iono` measures almost
   nothing.** Use near-mirror matchups to measure skill, and treat any cross-deck
   number as a matchup report.

---

## 3. The arena now has a real opponent: `rule:v10`

`scripts/import_v10_agent.py` lifts the LB-950 notebook into
`agents/agentkit/rulebased/sources/v10.py` (verbatim except the deck.csv
round-trip) plus `decks/lucario_v10.py` (its own retuned 60 — 4 Riolu, 3 Boss's
Orders, 14 energy, 2 Poké Pad, *not* `decks/mega_lucario_ex.py`). Idempotent;
re-run if the notebook updates.

```powershell
python -X utf8 scripts/arena.py play bc "rule:v10,noS" `
    --deck-a grimmsnarl --deck-b lucario_v10 --matches 500
```

Flags: `noS` disables its MCTS, `tb<sec>` sets its per-decision wall-clock
budget (default 1.5). In practice the flags do nothing, because the MCTS never
runs (§5) — `rule:v10` and `rule:v10,noS` are the same agent, and both cost
~0.15 s/game, as cheap as the clone. Keep passing `noS` anyway so the archived
agent name records the intent. `rule:v10x` is the variant with the search made
reachable (it still falls back to the policy; see §5).

`agentkit.rulebased.make_rule_agent(name, deck, overrides)` pokes module globals
after exec, so two configs of the same agent stay independent in one process.

---

## 4. The clone (what ships) — plateaued and now settled

`agents/sa/policy_net.npz` = **`policy_lw2`**, shipped as `55048039`.
Backups: `out/policy_net_bce_shipped.npz` (v1), `out/policy_lw3.npz`,
`out/policy_win.npz` (both rejected).

Head-to-head, n=2000, same deck both sides, vs the *previously shipped* net:

| net | corpus | val top-1 | vs prev shipped | ship? |
|---|---|---|---|---|
| v1 (BCE, 256/128) | 2,410 | 0.6596 | — | shipped `55046717` |
| listwise, 512,256/256,128 | 2,410 | 0.6711 | 0.514 [0.492, 0.536] | no |
| **v2 (`policy_lw2`)** | **2,810** | **0.6755** | **0.524 [0.502, 0.546]** | **YES — live** |
| `policy_lw3` (+07-17/18/19) | 4,010 | 0.6933 | 0.491 [0.469, 0.513] | no |
| `policy_win` (`--winners-only`) | 2,810 | 0.6410 | **0.375 [0.354, 0.396]** | **no — decisive** |

**Three axes are now settled negative on this net: more data, more accuracy, and
winners-only.** `--winners-only` is not marginal, it is 12pp worse — cloning the
losing side's moves is *helping* (they are mostly the same good moves, and it
doubles the data). Do not revisit. The clone has plateaued.

### Trainer

`scripts/train_policy.py` — `--loss listwise` is the right objective. Layers
export generically (`sfc{i}_w`/`head{i}_w` + counts), so depth changes need no
inference edit. Val plateaus by epoch ~5–10; ~12 epochs is plenty.
`artifacts/pds_v2/` is the frozen 2,810-game v2 corpus (`artifacts/pds` is now
4,010 and reproduces the *rejected* lw3).

### Deck choice: REOPENED — the sweep measured the wrong thing

The old sweep ranked decks by how well our clone beat `rule:iono`
(grimmsnarl 0.480 > alakazam 0.320 > … > mega_lucario_ex 0.030). Per rule 8 that
number is dominated by the deck-vs-iono matchup. Note that `mega_lucario_ex`
came *last* in that sweep, yet in rule-agent hands it beats the iono deck 0.78
and is the deck the LB-950 agent plays. **The sweep says which decks our clone
can pilot — nothing about which deck is strong.** Re-rank against `rule:v10`.

### Clone vs the field (grimmsnarl, seat-swapped)

| opponent | score | n |
|---|---|---|
| `rule:dragapult` | 0.519 [0.470, 0.567] | 400 |
| `rule:iono` | 0.480 [0.449, 0.511] | 1000 |
| `rule:lucario` | 0.475 [0.427, 0.524] | 400 |
| **`rule:v10,noS`** | **0.418 [0.388, 0.449]** | **1000** |
| `rule:abomasnow` | 0.360 [0.314, 0.408] | 400 |
| `random` | 0.995 | 200 |

---

## 5. Search: settled negative here AND in the field

**Ours:** `search:M,noV,roll,mo,mc20,pb0.15` vs `bc` = 0.323 [0.186, 0.499],
n=31. Measured cause: **the search overrules the clone on 52% of anchored
decisions.** A terminal rollout returns 0/1, so a mean over 12 determinizations
has SE ≈ 0.14; the max over ~9 rivals sits ~0.21–0.28 above its true value by
chance. Also negative: more determinizations (48 vs 12: 0.479, n=48) and the
value net (0.396, n=24). **Do not re-tune this.**

**V10 appears to do a smarter search — a shallow MCTS with a dense handcrafted
`evaluate_state` at a 1-turn horizon, which is exactly the low-variance leaf
evaluator our search lacks. It has never once executed.** Two independent bugs,
both verified by instrumenting the agent (`sim_calls` counters):

1. **The candidate set is always a single element.** `SEARCH_ALGO` builds it
   from `AdvancedPolicy.choose()`, which truncates to `select.maxCount` — and
   *every* MAIN select in this game has `maxCount == 1` (measured: 70/70 MAIN
   selects over 3 games, 2–27 options each, all maxCount 1). So `SEARCH_ALGO`
   takes its `len(candidates) == 1` early return every time and
   `simulate_action` is never called.
2. **Its `search_begin` call does not typecheck.** V10 calls
   `search_begin(obs, your_deck=yd)`; the SDK signature is
   `search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize,
   opponent_hand, opponent_active, manual_coin=False)`. It raises `TypeError`,
   which `SEARCH_ALGO`'s bare `except Exception: return None` swallows.

`rule:v10x` (same importer) fixes bug 1 so the search is reachable; bug 2 then
raises on every simulation, so it still falls back to the policy. Confirming
A/B, both ways: `rule:v10,tb0.2` vs `rule:v10,noS` = 0.492 [0.424, 0.561],
n=200, **11.8 s for 200 games** — the timing alone proves nothing ran.

**Therefore V10's LB 950+ is 100% handcrafted policy.** Nothing in this
competition has yet demonstrated that search is worth anything: ours measured
negative, and the field's best public example never ran. Treat search as
unproven, not promising.

The open question (not a blocker, and no longer the priority): fixing bug 2 is
easy for us — `agents/sa/worlds.py`'s `World` is *precisely* the `search_begin`
argument bundle, so `simulate_action` just needs a determinization from
`worlds.py` instead of `random.sample(my_deck, deckCount)`. That would give the
first real measurement of shallow-search-with-dense-eval in this game. Do it
only after P2 has a policy worth searching over.

---

## 6. Code map (`agents/sa/`)

- `bcagent.py` — **`PolicyAgent`, what we ship.** `net_path` pins a specific npz.
- `policynet.py` — numpy inference. `SA_PNET_PATH` env override; **dim guard**
  (stale net → `None` → fallback, never remove it).
- `features.py` (v2, DENSE_DIM=242, PER_SLOT=18) / `optfeat.py` — shared by
  trainers and inference. **Any npz trained pre-v2 fails the dim guard.**
- `agent.py` — `SearchAgent` (`main_only` knob). `planner.py` — determinized
  search (§5). `timemgr.py` — 600 s pool budgeting.
- `evalfn.py` + `textdmg.py` — handcrafted eval / expected damage. **Re-read
  these against V10's `evaluate_state`; they are the same object.**
- `worlds.py`, `tracker.py`, `fastsearch.py`, `deck_library.json`.
- Both agents never raise: fallback = `list(range(minCount))`.

---

## 7. Commands

```powershell
# Skill measurement: near-mirror head-to-head (rule 8). The only kind that counts.
python -X utf8 scripts/arena.py play "rule:v10,noS" rule:lucario `
    --deck-a lucario_v10 --deck-b mega_lucario_ex --matches 500

# Net A/B. ~5 min, n=2000.
python -X utf8 scripts/arena.py play "bc:new,net=out/policy_X.npz" bc `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 `
    --archive out/arena/ab_X.jsonl

# Against the real bar
python -X utf8 scripts/arena.py play bc "rule:v10,noS" `
    --deck-a grimmsnarl --deck-b lucario_v10 --matches 500

python -X utf8 scripts/tally.py "<agent-name>" "out/arena/foo_*.jsonl"

# Train (12 epochs; artifacts/pds_v2 is the shipped corpus)
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v2 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 --out out/policy_X.npz

# Import public notebook agents
python -X utf8 scripts/import_v10_agent.py     # rule:v10 + decks/lucario_v10
python -X utf8 scripts/import_rule_agents.py   # the four sample agents

# Build + submit (smoke-tests the bundle the way Kaggle loads it)
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc --nets policy
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_submit('dist/submission.tar.gz','msg','pokemon-tcg-ai-battle')"

# Public notebooks (this is how V10 was found — do it again periodically)
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(k.ref,'|',k.title) for k in a.kernels_list(competition='pokemon-tcg-ai-battle',sort_by='voteCount',page_size=30)]"
```

### Data on disk

`replays/`: 07-17..07-22, 07-24, 07-26, 07-27 = 400 each; 07-23 = 175, 07-25 =
268 (incomplete); 07-16 = 115; 07-13/14/15 = 0. Plus 366 old-repo replays at
`E:\Kaggle\pokemon-tcg-simulation\replay_miner\replays\2026-07-06..12`.
`artifacts/pds/` = 4,010 games (the rejected lw3 corpus);
`artifacts/pds_v2/` = 2,810 games (the shipped v2 corpus).
**§4 says more data is not the lever — do not spend time fetching.**

---

## 8. THE PLAN (day 4)

The strategy changed today. Two days of imitation learning produced an agent at
parity with a sample rule agent, ~200 below a public notebook. The field's
strongest work is hand-written domain logic. **Stop trying to learn the policy
and start writing it**, using the clone as a fallback for the decisions the
rules do not cover.

### P0 — DONE. Search is out (§5)

V10's MCTS never executes. Its 950 is pure handcrafted policy, so there is no
evidence anywhere that search helps in this game. All remaining effort goes into
the policy. (One loose end: `55049206`'s LB score is still moving — take one
more reading tomorrow to close the §1 table, then its slot is free.)

### P1 — Re-rank the decks against `rule:v10` (cheap, unblocks everything)

The deck sweep is invalid (§4). Re-run `scripts/deck_sweep.ps1` with
`rule:v10,noS` as the opponent instead of `rule:iono`, n≥400 each. Grimmsnarl
may well still win *for our clone*, but we need to know whether we are piloting
a weak deck before investing in the pilot. Cost: ~7 decks × 400 games × 0.3 s
≈ 15 min.

### P2 — Write a real agent for one deck (the actual path up)

This is where the leaderboard is, and after P0 it is the *only* thing on the
list that moves us. Pick the deck P1 endorses and write V10-style logic for it:
explicit attack planning (attacker/target/attack index with weakness and prize
arithmetic), per-option scoring, and matchup branches for the archetypes that
beat us. No search — §5.

**The bar to clear is 0.5 against `rule:v10,noS` in a near-mirror.** V10 is
~350 lines of readable scoring rules and it is worth ~950; that is the shape of
the thing to write, and it is a tractable amount of code.

Two concrete accelerants:
- **`rule:v10` is a working, readable template** for exactly this, and it is
  already in-process and measurable.
- **The clone is a good fallback**, and `bc` already answers every non-MAIN
  select competently. A hybrid — rules where we have them, clone elsewhere —
  starts strictly above whichever component is better on that decision class.

Measure every increment near-mirror against `rule:v10,noS` (rule 8), n≥500.

### P3 — The abomasnow hole (still open, still cheap)

0.360 vs 0.418–0.519 elsewhere, and our selects/turn collapse from 12.5–16.6 to
**8.6** with shorter games — a lockdown, not subtle misplay. Replay a loss with
`SA_DEBUG=1` and look at the actual select options on our turns. Fold the answer
into P2 as a matchup branch rather than fixing it in the net.

### Dropped

- **Self-play RL.** Days of work on 1.4 cores to maybe reach where hand-written
  rules already sit. The evidence that it is necessary evaporated: nothing at the
  top of this board is learned.
- **More replay data / more clone accuracy / `--winners-only`** — all settled
  negative (§4).

### Submission discipline

5 slots/day, **latest 2 active**. Submit only what has won head-to-head at
n≥500 against `rule:v10,noS`. Always `--nets` pin the config.
`55049206` (live `rule:iono`) is still converging — one more reading tomorrow
finishes P0; after that its slot is free.

---

## 9. Gotchas (all paid for)

- **`__file__` DOES NOT EXIST on Kaggle.** `kaggle_environments/agent.py` does
  `exec(code_object, env)` → `NameError` → Status=ERROR before the agent runs.
  This killed `55028078`. The smoke test `exec`s the source with no `__file__`
  in globals, exactly as Kaggle does — **keep it that way**.
- **Kaggle sets no env vars.** `SA_NO_PNET`/`SA_NO_VNET`/`SA_PNET_PATH` are inert
  there, so any bundled `.npz` is LIVE. Pin with `--nets none|policy|value|both`.
- **Do not set `SA_COUNT_MODE=expect` with a listwise-trained net.** It assumes
  calibrated probabilities; listwise gives a valid *ranking* only.
- **The harness does not enforce the 600 s pool but Kaggle does** (exhausted
  pool = loss). `arena.py` records `pool0`/`pool1` and warns below 300 s. BC uses
  0.1 s. V10 spends 1.5 s per MAIN decision — check the pool before copying that.
- **PowerShell `-File script.ps1 -Days a,b,c` does not bind an array.** Edit the
  script's default and launch with no args.
- **Never name a PowerShell param `$Matches`** — collides with the automatic
  regex variable.
- `kaggle competitions submit` may 400 even though the upload hit 100%; the
  Python client works. That call **submits** — it is not a dry run.
- Kaggle Python API returns **snake_case** fields (`public_score`, `team_name`),
  and `competition_leaderboard_view` paginates at 20 rows.
- Windows: `python -X utf8` everywhere. Run from the repo root; `sys.path` needs
  `src/`, `agents/`, root.
- Launch long jobs with `Start-Process` (detached); pass `-u` or python
  block-buffers the redirected stdout.
- Some replays download truncated (exactly 3 MiB) and fail JSON parse; builders
  skip them (`errors=N`). Delete + re-fetch to recover.
- Old repo `E:\Kaggle\pokemon-tcg-simulation` = failed pure-RL attempts. Take its
  replays, not its approach.
- Commit style: fine-grained, one-line semantic messages + Claude co-author
  trailer.

---

## 10. Superseded — do not resurrect

- **The old arena→LB ladder anchored on `rule:iono`.** Rule 8 killed it: two
  agents 104 Elo apart score identically against `rule:iono`. Any rating
  predicted through that anchor is meaningless.
- **The deck sweep's ranking** (§4) — measured deck-vs-iono matchup, not strength.
- **"The clone is comfortably above the rule baseline."** P0 says parity.
- **"The only unlock for search is a learned on-distribution value net."** And
  its successor, "V10 proves a handcrafted leaf eval works." V10's search has
  never executed (§5). Search is unproven in this competition, full stop.
- **All n=24 numbers**, and every strength claim dated before 2026-07-27 pm.
- **"3× compute made it worse"** — tested via `SA_SPEND_MULT`, which only grants
  time, and time was never binding. It measured nothing.
