# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** win the public LB (target 1200+; #1 "flg" = 1210, 5,792 teams).
Deadline **2026-08-16**, then ~2 weeks of continued play. Kaggle CLI is
authenticated and the user has entered. **This file must always end with a live
plan, never a summary.**

---

## 1. Where we are (end of day 2, 2026-07-28)

**The agent is the BC policy clone** (`agents/sa/bcagent.py`, ~1 ms/move). The
determinized search is measured *worse* than the clone and is not shipped.

Live leaderboard (all still converging — see the reading rule in §2):

| ref | what | public | state |
|---|---|---|---|
| `55049206` | **`rule:iono` LIVE baseline** | 629.0, rising | **active — do not evict** |
| `55048039` | **clone v2** (listwise, 2,810-game corpus) | 752.8, rising | **active** |
| `55046717` | clone v1 | 739.6 | frozen (evicted) |
| `55028156` | search + policy clone | 666.1 | frozen |
| `54647126` | `rule:iono` submitted 07-13 | 763.7 | **stale — see §2** |

**The single most important number to read next:** where `55049206`
(live `rule:iono`) settles. If it lands well below clone v2, we are *above* the
rule baseline and the old 763.7 was just an inflated number from a weaker July
pool. Early signs say yes (629 vs 753), and the local arena agrees (clone scores
0.480 vs `rule:iono` ≈ parity). **Do not submit anything until it converges** —
a new submission evicts it and kills the experiment.

### The day's real conclusion

Both compute-side levers are now settled negative, and the imitation lever has
plateaued:

- **Search does not work** (§4) — and we know exactly why, quantitatively.
- **Clone accuracy has decoupled from strength** (§3). Going 2,810 → 4,010
  training games raised val top-1 by +1.8pp and produced **zero** arena gain
  (0.491 [0.469, 0.513], n=2000).

So the cheap axes are exhausted. §7 is the plan; the honest headline is that
reaching 1210 now needs **self-play RL from the BC initialization**, because BC
cannot exceed its demonstrators and more imitation is no longer buying wins.

---

## 2. How not to fool yourself (read this before trusting any number)

This project has repeatedly drawn wrong conclusions from bad measurement. Every
rule below was paid for.

1. **n=24 is noise.** A BC game costs ~0.17 s, so n=1000 is 17 seconds of CPU.
   The project ran on n=24 for weeks; at n=1000 the "worse" agent turned out to
   be much better. **Never accept an n<100 strength claim for anything cheap to
   measure.** For a ~2pp effect you need n≈2000.
2. **A fresh LB score is not a result.** `55046717` read 865.2 → 697.9 → 739.7
   within two hours; `55028156` peaked ~925 and settled at 666.1. **Require two
   readings ≥1 h apart that agree.** And *frozen ≠ converged*: only the latest 2
   submissions play episodes, so an evicted submission's score just stops moving.
3. **Old submission scores are not comparable to new ones.** `rule:iono`'s 763.7
   was earned 07-13 against a smaller pool and frozen. That is why it is being
   re-measured live.
4. **Validation metrics do not predict playing strength here — four times now.**
   Value-net val loss (best net played worst), and policy top-1 three separate
   times. Judge every net in the arena, head-to-head, before shipping.
5. **Compare nets head-to-head, not through a third opponent.** `bc:<tag>,net=
   <path>` plays two nets in one process; going via `rule:iono` needs ~2x the
   games for the same resolution.
6. **CPU contention distorts `search` results but not `bc` results.** `search`
   budgets on wall-clock, so two search configs measured under different load
   are not comparable — use same-process head-to-head. BC agents are not
   time-budgeted, so cross-run BC comparisons are valid.
7. **This machine delivers ~1.4 cores of real throughput** (Ryzen 5500U, 15 W,
   plugged in, Balanced is the only power scheme). Running 4+ heavy jobs makes
   everything slower with no extra work done. Run 2–3. Prefer BC experiments
   (0.3 s/game) over search experiments (90–290 s/game) whenever a question can
   be posed either way.

---

## 3. The clone (what ships)

`agents/sa/policy_net.npz` = **`policy_lw2`**, shipped as `55048039`.
Backups: `out/policy_net_bce_shipped.npz` (v1), `out/policy_lw3.npz` (rejected).

All head-to-head, n=2000, same deck both sides, vs the *previously shipped* net:

| net | corpus | val top-1 | vs prev shipped | ship? |
|---|---|---|---|---|
| v1 (BCE, 256/128) | 2,410 | 0.6596 | — | shipped `55046717` |
| listwise, 512,256/256,128 | 2,410 | 0.6711 | 0.514 [0.492, 0.536] | **no** |
| **v2 (`policy_lw2`) = + 07-27** | **2,810** | **0.6755** | **0.524 [0.502, 0.546]** | **YES — live** |
| `policy_lw3` = + 07-17/18/19 | 4,010 | **0.6933** | 0.491 [0.469, 0.513] | **no** |

**Read the last two rows together — this is the important result.** +1.6pp of
val top-1 bought a marginal +2.4pp of win rate; the next +1.8pp bought
**nothing** (point estimate slightly negative, tight CI). The clone has
plateaued near 0.68–0.69 top-1, and additional imitation accuracy no longer
converts into wins. More replay data is *not* the answer any more.

(Caveat worth keeping in mind: v2's win had a lower bound of 0.502 — barely
separated. It is possible v2 ≈ v1 too and the whole listwise/depth/data
sequence has bought less than it appears.)

### Trainer

`scripts/train_policy.py` — `--loss listwise` (softmax CE within each select's
option set) is the right objective and reaches in 1 epoch what BCE took 4 to
reach. Layers export generically (`sfc{i}_w`/`head{i}_w` + counts), so depth
changes need no inference edit. Val plateaus by epoch ~5–10 while train loss
keeps falling — it overfits after that, so ~12 epochs is plenty.

Untried: `--winners-only` (we currently imitate the losing side's moves too) —
the last cheap idea on this axis, and worth one run.

### Deck choice: SETTLED — grimmsnarl. Stop re-testing.

Same clone, vs `rule:iono`: grimmsnarl **0.480** > alakazam 0.320 >
dragapult_ex 0.153 > crispin_box 0.068 > mega_abomasnow_ex 0.037 >
mega_lucario_ex 0.030 > iono 0.023. A deck's *meta* win-rate says nothing about
whether our clone can pilot it — `crispin_box` has the best meta WR (61.9%) and
scores 7%. Re-run `scripts/deck_sweep.ps1` only if the corpus changes a lot.

### Clone vs the rule field (grimmsnarl, seat-swapped)

| opponent | score | n |
|---|---|---|
| `rule:dragapult` | 0.519 [0.470, 0.567] | 400 |
| `rule:iono` | 0.480 [0.449, 0.511] | 1000 |
| `rule:lucario` | 0.475 [0.427, 0.524] | 400 |
| **`rule:abomasnow`** | **0.360 [0.314, 0.408]** | 400 |
| `random` | 0.995 | 200 |

---

## 4. Search: SETTLED NEGATIVE — do not re-tune it

`search:M,noV,roll,mo,mc20,pb0.15` vs `bc`: **0.323 [0.186, 0.499], n=31 —
significantly worse than the clone alone.**

**Why, measured not guessed: the search overrules the clone on 52% of anchored
decisions.** A terminal rollout returns 0/1, so a mean over 12 determinizations
has SE ≈ 0.14; the max over ~9 rival candidates sits ~0.21–0.28 above its true
value by chance, clearing any sane anchor margin. Half of all MAIN decisions
replaced the clone's judgment with a noisy tiebreak among its own top-10
candidates — and the clone is far better than that.

**This is a variance problem, not a tuning problem.** Raise the margin until
deviations are rare and you have reproduced the clone; lower it and you lose
harder. Real action differences are worth a few pp of win probability, and
resolving those with 0/1 rollouts needs ~100x more samples than the 600 s pool
buys (we already spend 92 s/game).

Also settled negative: **more determinizations** (48 vs 12: 0.479, n=48) and the
**value net** (0.396, n=24 — no evidence it helps, and it costs compute).

**The only unlock for search is a low-variance leaf evaluator.** The existing
value net was trained on *replay* states but queried at *search leaves* —
off-distribution, which is exactly why it failed. The correct object is a value
net trained on **states sampled from the search-leaf distribution, labeled by
rollout outcome**. Self-play RL (§7) would produce this as a by-product.

Implementation kept for that day: `roll` (rollout to terminal, 100% terminate,
~88–98 steps), `pb<margin>` (prior anchor), `mo` (MAIN-only — the clone answers
non-MAIN selects; cuts cost 288 s → 92 s per game), `HALVE_AFTER_WORLDS` (don't
prune on 1–2 noisy samples), and the `anchored`/`deviated` counters that proved
all this.

---

## 5. Code map (`agents/sa/`)

- `bcagent.py` — **`PolicyAgent`, what we ship.** `net_path` pins a specific npz.
- `policynet.py` — numpy inference. `SA_PNET_PATH` env override; **dim guard**
  (stale net → `None` → fallback, never remove it); `load()` is separable from
  the `get()` singleton so two nets can coexist.
- `features.py` (v2, DENSE_DIM=242, PER_SLOT=18) / `optfeat.py` — shared by
  trainers and inference. **Any npz trained pre-v2 fails the dim guard.**
- `agent.py` — `SearchAgent` (`main_only` knob). `planner.py` — determinized
  search (see §4). `timemgr.py` — 600 s pool budgeting, per-instance caps.
- `evalfn.py` + `textdmg.py` — handcrafted eval / expected damage.
- `worlds.py`, `tracker.py`, `fastsearch.py`, `deck_library.json` — determin-
  ization, cross-turn opponent-card tracking, raw-dict search wrapper.
- Both agents never raise: fallback = `list(range(minCount))`.

---

## 6. Commands

```powershell
# Head-to-head net A/B (the only comparison that counts). ~10-15 min, n=2000.
python -X utf8 scripts/arena.py play "bc:new,net=out/policy_X.npz" bc `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 `
    --archive out/arena/ab_X.jsonl

# Absolute score vs the field
python -X utf8 scripts/arena.py play bc rule:iono --deck-a grimmsnarl --deck-b iono --matches 500

# Pool sharded runs into one Wilson interval
python -X utf8 scripts/tally.py "<agent-name>" "out/arena/foo_*.jsonl"

# Train (12 epochs is plenty; it overfits after ~5-10)
python -X utf8 scripts/train_policy.py --ds artifacts/pds --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 --out out/policy_X.npz

# Data
python -X utf8 scripts/fetch_top_episodes.py --date 2026-07-16 --max 400   # idempotent
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds/d16 replays/2026-07-16

# Build + submit (smoke-tests the extracted bundle the way Kaggle loads it)
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc --nets policy
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_submit('dist/submission.tar.gz','msg','pokemon-tcg-ai-battle')"
```

Batch helpers: `bc_sweep.ps1`, `deck_sweep.ps1`, `fetch_days.ps1`,
`build_days.ps1` (edit the `$Days`/`$Dates` default and launch with **no**
arguments — see Gotchas).

### Data on disk

`replays/`: 07-17..07-22, 07-24, 07-26, 07-27 = 400 each; **07-23 = 175 and
07-25 = 268 (incomplete — re-fetch)**; 07-16 = 115 and 07-13/14/15 = 0 (that
fetch was interrupted; `fetch_days.ps1` is idempotent, just re-run it — though
§3 says more data is no longer the lever). Plus 366 old-repo replays at
`E:\Kaggle\pokemon-tcg-simulation\replay_miner\replays\2026-07-06..12`.
`artifacts/pds/`: `old, d17, d18, d19, d21..d27` → 4,010 games / 607k rows.
Each daily manifest holds ~4,400–4,800 episodes and we take only the top 400
(cutoff avg_score ~1174–1205), so `--max 800` would roughly double the corpus —
**but see §3: more data stopped helping.**

---

## 7. THE PLAN (day 3)

Ordered by expected value. The first item is free and blocking.

### P0 — Read the live baseline, then decide (blocking, costs nothing)

Wait for `55049206` (live `rule:iono`) and `55048039` (clone v2) to converge;
two readings ≥1 h apart. This resolves the biggest open question in the project:
**are we above or below the real rule-based bar?** Do not submit anything until
it lands — a submission evicts the baseline. Then re-fit the ladder
(rating ≈ base + 400·log10(S/(1−S))) against *live* numbers; the old ladder,
anchored on the stale 763.7, over-predicted us by ~50 points.

### P1 — The abomasnow hole (best remaining cheap win)

0.360 vs 0.475–0.519 everywhere else, and the ladder over-predicted our rating
in exactly the way a matchup hole explains. **First diagnostic is already done
and it points at a lockdown, not subtle misplay** — our selects per turn:

| opponent | selects/turn | avg turns (W/L) |
|---|---|---|
| `rule:iono` | 16.6 | 12.4 / 13.2 |
| `rule:dragapult` | 12.9 | 13.1 / 13.7 |
| `rule:lucario` | 12.5 | 12.3 / 11.8 |
| **`rule:abomasnow`** | **8.6** | **10.3 / 12.4** |

We take roughly *half* as many actions per turn and the games are shorter.
Something stops us acting — item/ability lock, status, or failure to develop.
Next: replay a loss with `SA_DEBUG=1` and look at the actual select options on
our turns. Then ask whether the corpus even *contains* grimmsnarl-vs-abomasnow
games (`scripts/mine_meta.py`); if not, no amount of general data fixes it and
targeted replays are needed.

### P2 — `--winners-only` (last cheap clone idea)

One training run + one n=2000 A/B, ~40 min total. We currently clone the losing
side's moves as well as the winner's. Low odds given §3, but cheap.

### P3 — Self-play RL from the BC initialization (the only path to 1210)

BC cannot exceed its demonstrators by construction, and imitation accuracy has
stopped converting into wins (§3). This is the standard next step and the one
thing on this list that can actually reach the target. It also produces, as a
by-product, the on-distribution value function that would revive search (§4).

Scope it honestly before starting: it is days of work, the engine does ~2,600
search-steps/s, and **this machine gives ~1.4 cores**. Consider whether to rent
compute. Start from the shipped clone as both policy init and opponent pool.

### Submission discipline

5 slots/day, **latest 2 active**. Submit only what has already won head-to-head
at n≥2000. Always `--nets` pin the config. Never submit an unmeasured build.
Right now both active slots are spoken for by the P0 experiment.

---

## 8. Gotchas (all paid for)

- **`__file__` DOES NOT EXIST on Kaggle.** `kaggle_environments/agent.py` does
  `exec(code_object, env)` → `NameError` → Status=ERROR before the agent runs.
  This killed `55028078`. `main.py` resolves its dir via try/except NameError →
  `/kaggle_simulations/agent` → cwd. The smoke test now `exec`s the source with
  no `__file__` in globals, exactly as Kaggle does — **keep it that way**; the
  old `import main` smoke defined `__file__` and hid the bug.
- **Kaggle sets no env vars.** `SA_NO_PNET`/`SA_NO_VNET`/`SA_PNET_PATH` are all
  inert there, so any bundled `.npz` is LIVE. Pin with `build_submission.py
  --nets none|policy|value|both`.
- **Do not set `SA_COUNT_MODE=expect` with a listwise-trained net.** `expect`
  picks the multi-select count by summing per-option sigmoids, which assumes
  calibrated probabilities; a listwise net gives a valid *ranking* only. The
  default `table` mode is loss-independent and correct for every net we have.
- **The harness does not enforce the 600 s pool but Kaggle does** (exhausted
  pool = loss). `arena.py` records `pool0`/`pool1` per game and warns below
  300 s. Check it before shipping anything that searches. BC uses 0.1 s.
- **PowerShell `-File script.ps1 -Days a,b,c` does not bind an array.**
  Space-separated spills onto the *next* positional param (silently making every
  day look MISSING); comma-joined arrives as one string. Edit the script's
  default and launch with no args.
- **Never name a PowerShell param `$Matches`** — collides with the automatic
  regex variable; every assignment throws `ArgumentTransformationMetadata`.
- `kaggle competitions submit` may 400 even though the upload hit 100%; the
  Python client works. That call **submits** — it is not a dry run.
- Kaggle Python API returns `ApiSubmission` with **snake_case** fields
  (`public_score`, not `publicScore`).
- Windows: `python -X utf8` everywhere (cp1252 crashes on card names). Run from
  the repo root; `sys.path` needs `src/`, `agents/`, root.
- Launch long jobs with `Start-Process` (detached); bash `nohup &` dies with the
  session. Redirecting python stdout block-buffers it — pass `-u`.
- Some replays download truncated (exactly 3 MiB) and fail JSON parse; builders
  skip them (`errors=N`). Delete + re-fetch to recover.
- Old repo `E:\Kaggle\pokemon-tcg-simulation` = failed pure-RL attempts. Take
  its replays, not its approach.
- Commit style: fine-grained, one-line semantic messages + Claude co-author
  trailer.

---

## 9. Superseded — do not resurrect

- **Every strength claim in this file dated before 2026-07-27 pm** was measured
  through broken controls (nets silently rejected by dim guards, a compute knob
  that could not bind, a mirror matchup compared against cross-deck runs).
- **All n=24 numbers**, including "search+policy scores 0.33" — at n=1000 the
  clone alone scored 0.480.
- **The old arena→LB ladder anchored on `rule:iono` = 763.7.** Stale pool; it
  over-predicted. Re-fit after P0.
- **"3× compute made it worse"** — tested via `SA_SPEND_MULT`, which only grants
  time, and time was never binding (`MAX_WORLDS` was). It measured nothing.
