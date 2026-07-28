# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** win the public LB (target 1200+ Elo; current #1 "flg" = 1210, 5792 teams).
Deadline 2026-08-16, then ~2 weeks of continued play. User gives full token budget
for ~2 days (day 1 was 2026-07-27). Kaggle CLI is authenticated and the user has
entered the competition. **We are not stopping until we win — this file should
always end with a live plan, never a summary.**

**Status 2026-07-28 (day 2):** the search stack is not what we thought. Measured
at **n=1000** instead of n=24, the **BC clone alone scores 0.480 vs `rule:iono`**
— roughly LB parity with the 763.7 baseline, and far above the search agent's
0.33 (n=24) that produced our live 666.1. **Submitted `55046717` = the BC agent.**
The day's work has moved from "fix the search" to "make the clone better",
because the clone is both stronger and ~1000x cheaper to measure.
Read "Day-2 results" then "THE PLAN".

## Competition hard facts

- Submission: **.tar.gz** with `main.py` + `deck.csv` at **top level** + `cg/` engine.
  Cap 197.7 MiB. 5/day, latest 2 active. Runs at `/kaggle_simulations/agent/`.
- Runtime: 2 vCPUs, 12.2 GiB RAM. `actTimeout=0`; each agent has a **600s
  thinking pool per game** (`obs["remainingOverageTime"]`); exceed → timeout loss.
  Validation episode = self-play vs itself first (crash ⇒ Error status).
- Rating: TrueSkill-ish, new submissions start μ=600. Episodes vs similar-rated.
- Engine: `cg` (C++ DLL + Python ctypes). Search API (`search_begin`/`search_step`)
  supports branching tree search over determinized worlds (opponent hand visible
  in search), ~2600 steps/s. Local engine at `data/sample_submission/sample_submission/`.

## Approach (current architecture, all in `agents/sa/`)

Determinized turn-level search + learned nets (BC from top replays):

- `fastsearch.py` — raw-dict wrapper over engine search API (bypasses slow dataclasses).
- `worlds.py` — determinization; opponent decklist predicted from `deck_library.json`
  (built from mined top replays via `scripts/build_deck_library.py`).
- `tracker.py` — cross-turn log tracking (known cards in opponent hand by serial).
- `evalfn.py` + `textdmg.py` — handcrafted eval; text-parsed expected damage
  (bench-aware threat). Used for playout greedy fallback + leaf fallback.
- `features.py` — **v2** state featurization (DENSE_DIM=242, PER_SLOT=18) shared
  by trainers and numpy inference. v2 added attack-readiness + own-type energy count.
- `valuenet.py` / `policynet.py` — numpy-only inference (no torch in bundle) of
  `value_net.npz` / `policy_net.npz`; both have **dim guards** (stale net → None → fallback).
- `optfeat.py` — **v2** per-option features for policy (returns dense, card_id,
  attack_id, target_id; target = attach/evolve destination).
- `planner.py` — per-decision: determinize N worlds (≤12), root candidates
  (`candidate_combos`, policy-pruned to ~10), playout = policy-argmax for BOTH
  sides until our next MAIN (1.5 turns), leaf = value net (or evaluate() if no net),
  successive halving of root candidates across worlds. Env knobs: `SA_DEBUG`,
  `SA_NO_VNET`, `SA_NO_PNET`, `SA_SPEND_MULT`, `SA_MAIN_CAP`, `SA_MINOR_CAP`.
- `timemgr.py` — budgets the 600s pool (reserve 45s, mains ≤4.5s, scaled by branching).
- `agent.py` — `SearchAgent` (hybrid), `bcagent.py` — `PolicyAgent` (BC-only, ~ms/move).
  Both never raise: fallback = `list(range(minCount))`.

## Experimental results (local arena, seat-swapped) — REVISED 2026-07-27 (day 1 pm)

### Correction: earlier numbers in this file were measured through broken controls

The pre-2026-07-27 results above/below were invalid for three separate reasons.
**Do not trust any strength claim in this file dated before 2026-07-27 pm.**

1. Nets were trained pre-v2 (DENSE_DIM=218 vs 242) so **both dim guards silently
   rejected them**. Every "hybrid" result was really search+handcrafted-eval.
2. "3× compute made it worse (17%)" was tested via `SA_SPEND_MULT`, which only
   grants more *time*. Time was never binding — `MAX_WORLDS=12` was. That
   experiment changed nothing and measured noise.
3. The "50% search+handcrafted" figure was a **mirror** matchup (iono vs iono);
   it was then compared against cross-deck (grimmsnarl vs iono) runs. Measuring
   the same no-nets config cross-deck now gives **21%**, not 50%.

### Current measured truth (nets = full 6-day corpus)

vs `rule:iono` (grimmsnarl vs iono, n=24 each, wall-clock-contention caveat below):
- search, **both nets off** (pure handcrafted): **21%** (5/24)
- search, **policy ON**, value off: **33%** (8/24)  ← best config
- BC-only (policy alone, no search): **25%**

Head-to-head A/Bs (same process, same deck both sides — immune to CPU contention,
the only fully trustworthy comparisons):
- **policy ON vs policy OFF: 0.696 [0.567, 0.801], W38/D2/L16 over 56 games.**
  CI excludes 0.5 → the policy net is a real improvement. Settled.
  (Two independent 28-game runs of the same config ran concurrently by accident:
  0.750 [0.566,0.873] and 0.643 [0.458,0.793]. The second alone would NOT have
  been significant — 28 games is not enough to call a ~15pp effect. Always
  combine, and treat any single 24–28 game arena number as barely indicative.)
- **value ON vs value OFF: 0.396 [0.228, 0.592], W9/D1/L14 over 24 games.**
  CI *includes* 0.5, so this does NOT prove the value net hurts — it proves there
  is **no evidence it helps**, with a negative point estimate. Since it also costs
  compute per leaf, ship without it. (Tested with the *good* full-corpus net,
  val 0.5878 — i.e. the net that improved most on paper is the one we exclude.)
- **worlds 48 vs 12 — SETTLED NEGATIVE. Do not retry this axis.**
  | variant | score | n | CI |
  |---|---|---|---|
  | handcrafted leaf (arena5) | 0.500 | 24 | [0.314, 0.686] |
  | both nets on (arena7) | 0.458 | 24 | [0.279, 0.649] |
  | **combined** | **0.479** | **48** | **[0.345, 0.617]** |
  4× the determinizations buys nothing, with or without nets. It also does not
  hurt → the old "3× compute made it worse" claim is refuted (that test used
  `SA_SPEND_MULT`, which could not bind while `MAX_WORLDS` did).
  **Interpretation:** 12 determinizations already average out most hidden-info
  variance; more worlds cut *variance* but cannot fix a *biased* leaf evaluator.
  The 3%-of-budget headroom is real but is NOT free upside on this axis.
  If spending the budget, try a *different* axis: `MAX_PLAYOUT_STEPS` (rollout
  depth / horizon), `ROOT_CAP` (root width), or `PLAYOUT_CAP` — i.e. the
  quality/depth of each rollout, not the number of worlds.

### Two structural lessons

**Value net: val loss does NOT predict arena strength.** The net is trained on
states from top players' *real* games but queried at *search leaves* mid-playout
(after greedy/policy continuations). Those are off-distribution, so replay
accuracy does not transfer. Judge the value net by arena results only, never by
val loss. It now clearly beats logreg (0.5878 vs 0.634) and still appears to
make play worse.

**Data volume drives net quality, and it kept paying:**
| corpus | policy top-1 | value best val |
|---|---|---|
| 768 games (v1) | 57% | — |
| 1,211 / 1,188 games | 63.4% | 0.6327 |
| **2,410 / 2,387 games** | **66.0%** | **0.5878** |
Value net still peaks at *epoch 0* even at 2.4k games → still data-starved.
Late-game acc went 0.396 (noise, only 18 val games in that bucket) → 0.608.

### The biggest open lever: the agent uses 3% of its compute

`MAX_WORLDS` binds on **every** decision. Measured with `SA_DEBUG` instrumentation:
| MAX_WORLDS | worlds/decide | per-decision budget used | 600s pool used |
|---|---|---|---|
| 12 (default) | 12.0 | 15% | ~20s |
| 48 | 47.5 | 42% | ~89s |
Raising it is free headroom with no timeout risk. Whether it *helps* is unresolved
(arena5/arena7). Note arena5 runs a handcrafted leaf, where more determinizations
may just amplify eval bias — arena7 retests with nets on.

## Day-2 results (2026-07-28) — measure BC at n=1000, not n=24

### The headline: the clone beats the search stack

| agent | score vs `rule:iono` | n | implied LB |
|---|---|---|---|
| **`bc` (policy clone, ~1ms/move)** | **0.480 [0.449, 0.511]** | **1000** | **~750** |
| `search:pol,noV` | 0.33 [0.18, 0.52] | 24 | 666 (observed) |

The n=24 arena numbers this project ran on were **noise**. A BC agent plays a
game in ~0.17s, so 1000 games cost 17 seconds of CPU — there was never a reason
to measure it at n=24. Do not accept an n<100 strength claim again for anything
that can be measured cheaply.

**Cross-run comparison is valid for BC agents.** The contention caveat below
applies to `search` (wall-clock time budgets); a BC agent is not time-budgeted,
so two BC configs measured in separate processes against the same opponent are
directly comparable. This is what makes the fast loop possible.

### BC vs the whole rule-based field (grimmsnarl, seat-swapped)

| opponent | score | n |
|---|---|---|
| `rule:dragapult` | 0.519 [0.470, 0.567] | 400 |
| `rule:iono` | 0.480 [0.449, 0.511] | 1000 |
| `rule:lucario` | 0.475 [0.427, 0.524] | 400 |
| **`rule:abomasnow`** | **0.360 [0.314, 0.408]** | 400 |

Roughly parity with the field, with **abomasnow a real, measured weakness** —
the one matchup that is significantly below the rest. Worth a targeted look.

### Deck choice is NOT free — the clone can only pilot grimmsnarl

Same agent, same opponent (`rule:iono`), different deck:

| our deck | score | n |
|---|---|---|
| **`grimmsnarl`** | **0.480 [0.449, 0.511]** | 1000 |
| `alakazam` | 0.320 [0.276, 0.367] | 400 |
| `dragapult_ex` | 0.153 [0.117, 0.198] | 300 |
| `crispin_box` | 0.068 [0.047, 0.096] | 400 |
| `mega_abomasnow_ex` | 0.037 [0.021, 0.064] | 300 |
| `mega_lucario_ex` | 0.030 [0.016, 0.056] | 300 |
| `iono` (mirror) | 0.023 [0.011, 0.047] | 300 |

`crispin_box` has the best raw win-rate in the mined meta (61.9%) and our agent
scores **7%** with it. A deck's meta win-rate says nothing about whether *our*
policy can pilot it — the clone has to have learned that deck's lines, and the
clone has only really learned grimmsnarl. **Grimmsnarl is settled; stop
re-testing this.** Corollary: if the corpus ever shifts, re-run `deck_sweep.ps1`
(it costs ~10 min) before assuming grimmsnarl is still right.

### Policy training: the loss function was wrong

The old trainer used pointwise BCE (each option scored independently), which
does not model "which of these options is best" — the thing the agent actually
does. `--loss listwise` (softmax cross-entropy within each select's option set)
reaches **0.6216 val top-1 after ONE epoch**; BCE needed ~4 epochs to get there
and finished at 0.6596 after 20. The net was also underfit (still improving at
epoch 19), so it is now deeper as well (`--state-h 512,256 --head-h 256,128`).
Layers export generically, so depth changes need no inference edit.

### Terminal rollouts work mechanically but were deprioritized

`search:...,roll` plays the policy to game end and scores the real result
(`agents/sa/planner.py`). Verified: **100% of rollouts reach terminal**, ~98
steps each, ~156s of the 600s pool per game. It is implemented and ready.
It was **not** run to a conclusion: it costs ~150s/game to measure, and its
premise ("fix the leaf and search becomes good") was written when BC was
believed to score 0.25. BC scoring 0.480 inverts that premise, so the CPU went
to the clone instead. Re-run it as `search:R,noV,roll,mc12,nc6` vs `bc`.

## Leaderboard feedback — the arena is CALIBRATED (2026-07-28)

`55028156` settled at **666.1** (peaked ~925 mid-convergence, then fell back).
Our own `rule:iono` submission (`54647126`) sits at **763.7** on the same LB.

**Ignore the 925.** A new submission starts at μ=600 with a wide σ, so μ swings
hard on the first few episodes. The peak during convergence is not a strength
estimate; the settled value is. Do not chase it, and do not report it as a result.

**The important finding is that the local arena predicted the LB.**
- Local arena: search+policy scores **0.33** vs `rule:iono` (n=24).
- Implied rating gap: `400·log10(0.33/0.67)` = **−123** → predicts **641**.
- Observed gap: 666.1 − 763.7 = **−98** → implies a **0.363** score.
Arena said 0.33, LB says 0.363 — agreement well inside the n=24 CI.

So `arena.py play <cfg> rule:iono --deck-a grimmsnarl --deck-b iono` is a **free,
unlimited, same-day proxy for the leaderboard**. This is the single most valuable
thing we learned from submitting. Stop spending submission slots to find out
whether a config is good; measure locally, submit only what already won.

**The ladder to the target** (rating = 763.7 + 400·log10(S/(1−S)), S = local score
vs `rule:iono`):
| local S vs rule:iono | implied LB rating |
|---|---|
| 0.36 ← **we are here** | 666 |
| 0.50 | 764 |
| 0.70 | 911 |
| 0.80 | 1004 |
| 0.93 | **1213 (wins the LB)** |

Read that last row carefully: **the #1 agent would beat `rule:iono` ~93% of the
time.** We are at 36% against an opponent whose source we can read. This is not a
tuning gap — treat it as an architecture gap.

Caveats, so nobody over-trusts this: TrueSkill μ is not exactly Elo-scaled, the LB
pool is not `rule:iono`, 666.1 may not be fully converged, and this is a **single**
calibration point. Budget ±100 on the ladder and re-check it after the next
submission lands.

## Data pipeline

- `scripts/fetch_top_episodes.py --date YYYY-MM-DD --max N` — downloads manifest.csv
  of daily dataset `kaggle/pokemon-tcg-ai-battle-episodes-<date>`, then top-N episodes
  by avg_score into `replays/<date>/`. Idempotent (skips existing). Now parallel (4 workers).
- Old repo has 366 top-1% replays: `E:\Kaggle\pokemon-tcg-simulation\replay_miner\replays\2026-07-06..12`.
- Downloaded (COMPLETE as of 2026-07-27 pm): `2026-07-26` (403), `07-25` (268),
  `07-24` (401), `07-23` (176), `07-22` (401), `07-21` (401), `07-20` (~400).
  Plus the old repo's 366. Datasets built for all: `artifacts/{ds,pds}/{old,d21..d26}`.
  → value 2,387 games / 390k rows; policy 2,410 games / 363k rows.
- A few replays download truncated (exactly 3 MiB) and fail JSON parse; the
  builders skip them (`errors=N` in the summary). Delete + re-fetch to recover.
- `scripts/build_dataset.py --out artifacts/ds/<tag> --stride 1 <dirs>` — value shards.
- `scripts/build_policy_dataset.py --out artifacts/pds/<tag> <dirs>` — policy shards.
- `scripts/train_value.py --ds artifacts/ds --epochs 4` → `agents/sa/value_net.npz`.
- `scripts/train_policy.py --ds artifacts/pds --epochs 6 [--winners-only]` →
  `agents/sa/policy_net.npz` (includes count_frac table for multi-select counts).
- **IMPORTANT:** features are v2 now; any npz trained pre-v2 fails the dim guard
  (agent silently falls back). Always rebuild datasets + retrain after feature changes.
- A background job was mid-run rebuilding v2 datasets (old+d26+d25) and retraining
  both nets. If dead, re-run:
  ```
  OLD="E:\Kaggle\pokemon-tcg-simulation\replay_miner\replays"
  python -X utf8 scripts/build_dataset.py --out artifacts/ds/old --stride 1 "$OLD/2026-07-06" ... "$OLD/2026-07-12"
  python -X utf8 scripts/build_dataset.py --out artifacts/ds/d26 --stride 1 replays/2026-07-26   (etc. per day)
  python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds/<tag> <dirs>
  python -X utf8 scripts/train_policy.py --ds artifacts/pds --epochs 6
  python -X utf8 scripts/train_value.py --ds artifacts/ds --epochs 4
  ```

## Meta / deck choice (from mining 2026-07-26 top episodes)

- #1 usage: **Marnie's Grimmsnarl ex / Munkidori** (flg, Dries, Luca…) → `decks/grimmsnarl.py`.
- Best WR (61.9%): **Crispin multi-energy box** (James Cox #2 @1218) → `decks/crispin_box.py`.
- Alakazam (Yushin Ito) → `decks/alakazam.py`. Sample decks: iono/dragapult_ex/mega_abomasnow_ex/mega_lucario_ex.
- External paper (papers/From Rules to Nash Equilibria.pdf) independently supports
  Grimmsnarl being under-played relative to strength. Default deck: **grimmsnarl**.
- `scripts/mine_meta.py <replay_dirs>` prints archetypes/teams/exact lists.
- `scripts/build_deck_library.py <replay_dirs>` regenerates `agents/sa/deck_library.json`
  (18 decks currently; rebuild when new days land).

## Testing

- `python -X utf8 scripts/arena.py play <specA> <specB> --deck-a X --deck-b Y --matches N`
  Specs: `search[:tag]`, `bc[:tag]`, `rule:iono|dragapult|abomasnow|lucario`, `random`.
  Deck specs = `decks/` module names. `scripts/arena.py elo` fits Elo over archive.
- Engine prints "No Basic Pokemon." to stdout on some determinization rejects — filter it.
- `python -X utf8 scripts/sdk_smoke.py` — engine sanity.
- `python -X utf8 scripts/probe_search_api.py` — search API sanity/throughput.

## Submission

- `python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent search
  --nets policy` → builds + **smoke-tests** `dist/submission_*.tar.gz`.
  `--nets` pins which npz ship (see Gotchas — this is load-bearing on Kaggle).
- **SUBMITTED 2026-07-27:**
  - `55028078` — **ERROR** (`__file__` NameError under Kaggle's exec loader; see Gotchas).
  - `55028156` — **COMPLETE, settled at 666.1** (peak ~925 during convergence —
    noise, see "Leaderboard feedback"). Config = policy ON / value OFF / worlds 12
    (the arena-winning config). **Linux env is VALIDATED — the bundle runs on Kaggle.**
- Full submission history (`competition_submissions`), for baselines:
  | ref | date | score | what |
  |---|---|---|---|
  | **55046717** | **07-28** | *pending* | **`bc` policy clone, grimmsnarl (current)** |
  | 55028156 | 07-27 | **666.1** | search + policy clone, grimmsnarl |
  | 54848951 | 07-20 | 477.1 | old-repo attempt |
  | 54727521 | 07-15 | 435.8 | ismcts |
  | **54647126** | 07-13 | **763.7** | **`rule:iono` — the bar to clear** |
  | 54535698 | 07-10 | 516.1 | Kaggle starter RL |
  | 54356986 | 07-05 | 361.4 | early attempt |
  Nothing we have built has ever beaten the stock rule agent. That is the headline.
- `kaggle competitions submit` returned a **400** on CreateSubmission even though
  the upload hit 100%; the identical file went through minutes later via the
  Python client. If the CLI 400s, use:
  `python -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi();
  a.authenticate(); a.competition_submit('dist/submission.tar.gz','msg','pokemon-tcg-ai-battle')"`
  — and note that call SUBMITS; it is not a dry run.
- Smoke caveat: the smoke opponent is trivial, so games end fast (~11s of pool).
  It proves the bundle imports and runs, not that a long game stays in budget.
  Arena data covers that: ~109 selects × 423ms ≈ 46s vs a 600s pool (13× headroom).

## THE PLAN (live — day 2 evening, 2026-07-28)

Framing has changed. The clone, not the search, is the agent. It sits at ~LB
parity with `rule:iono` (0.480) and we need 0.93 to win. **The clone's ceiling
is the players it imitates — and those players are rated 1170–1210, i.e. the
target.** So "make the clone imitate them better" is a path that actually
reaches the goal, and it iterates in minutes instead of hours.

### N0 — Finish and ship the better clone (in flight)

1. `out/policy_lw.npz` is training (listwise, 512/256 + 256/128, 30 epochs).
   Evaluate with
   `SA_PNET_PATH=out/policy_lw.npz` + `arena.py play bc rule:iono --deck-a
   grimmsnarl --deck-b iono --matches 500`. **Beat 0.480 → ship it.**
   (`SA_PNET_PATH` scores a candidate net without overwriting the shipped one.)
2. Then retrain on the enlarged corpus (07-27 is downloading; 07-19/18/17 queued)
   — more data has paid every single time (57 → 63.4 → 66.0% top-1).
3. Next trainer knobs to try, in order: `--winners-only` (clone only the winner's
   moves), `--loss both`, more epochs (it was still improving at 30).

### N1 — Re-test search ON TOP of the good clone

Do not conclude "search is worthless", only "this search, with this leaf, at
n=24, looked worse than the clone". The rollout leaf is now implemented and
verified (100% of rollouts terminate). The clean test is
`search:R,noV,roll,mc12,nc6` vs `bc`, same deck, n≥50. Note the #1 player is
reportedly "model + search using the full budget" — so search is not a dead end
in principle, and we currently use 0.1s of a 600s pool.

### N2 — The abomasnow hole

BC scores 0.360 vs `rule:abomasnow` but 0.48–0.52 vs everything else. One
matchup this far below the rest is a real, addressable weakness, and the LB pool
contains such decks. Diagnose before tuning: dump a few losses and look.

### N3 — Deck choice is settled: grimmsnarl

Do not spend more time here. Measured, same agent, vs `rule:iono`: grimmsnarl
0.480 > alakazam 0.320 >> crispin_box 0.068 > iono 0.023. The clone can only
pilot decks it has actually seen in the top replays.

### Submission discipline (changed as of today)

5 slots/day, 2 active, and **the arena now predicts the LB** — so slots are no
longer how we learn things. Rules: submit only configs that already beat the
current champion locally at n≥100; keep one slot to re-verify calibration; always
`--nets` pin the config; never submit an unmeasured build.

## Older next-steps (superseded 2026-07-28, kept for rationale)

- ~~Finish the compute question (worlds 48 vs 12)~~ — **settled negative**, see above.
- The agent loses to `rule:iono` in every config (21–33%). A 66%-accurate clone
  did not fix the prize race → suspect `evalfn.py`/`textdmg.py` or root candidate
  enumeration. (This became P0/P1.)

## Gotchas

- **`__file__` DOES NOT EXIST on Kaggle.** `kaggle_environments/agent.py` runs
  main.py via `exec(code_object, env)`, so `__file__` is undefined →
  `NameError` → submission Status=ERROR before the agent ever runs. This killed
  submission 55028078 (2026-07-27). `main.py` now resolves its dir via
  try/except NameError → `/kaggle_simulations/agent` → cwd.
  **The old smoke test could not catch this** because it did `import main`, and
  importing defines `__file__`. The smoke now `exec`s the source with no
  `__file__` in globals, exactly as Kaggle does. Keep it that way — a local
  smoke that loads the agent differently from the grader proves nothing.
- **Arena archiving is NOT incremental.** `arena.py` buffers rows and writes them
  in a `finally` at the END of the run. `Stop-Process -Force` skips the `finally`
  → **all games from that run are lost**. Let arenas finish, or stop them gently.
- **Arena strength numbers depend on machine load.** `timemgr` budgets on
  wall-clock, so a contended CPU searches less deeply. Two agents measured under
  different background load are NOT comparable. Head-to-head runs (both agents in
  one process) are immune — prefer them for every A/B.
- Launch long jobs via PowerShell `Start-Process` (detached); bash `nohup &` jobs
  die when the session's process group is torn down. There is no `setsid` here.
- Redirecting python stdout to a file block-buffers it; pass `-u` or output only
  appears in chunks. Never pipe a long run through `| grep | tail` — that buffers
  until exit, so a killed run looks like it produced nothing (this cost a run).
- **Kaggle sets no env vars**, so `SA_NO_PNET`/`SA_NO_VNET` are 0 there and any
  bundled `.npz` is LIVE. Pin the config with `build_submission.py --nets
  none|policy|value|both` (omitting an npz → `get()` returns None → fallback).
- Per-instance knobs (added 2026-07-27) let two configs be A/B'd in one process:
  `search:<tag>,noP,noV,w<N>` in arena specs; `SearchAgent(deck, no_pnet=,
  no_vnet=, max_worlds=)`. The module-level `SA_*` env vars apply to BOTH sides
  and would silently compare two identical configs.
- Windows: use `python -X utf8` everywhere (cp1252 crashes on card names).
- Arena/harness must run from repo root; `sys.path` needs `src/`, `agents/`, root.
- Background bash jobs die with the session; re-launch fetches idempotently.
- Old repo `E:\Kaggle\pokemon-tcg-simulation` = failed pure-RL attempts; don't import
  its approach, only its replays. gitignore keeps data/replays/artifacts/dist out of git.
- Commit style: fine-grained, one-line semantic messages + Claude co-author trailer.
