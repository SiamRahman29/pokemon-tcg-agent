# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** win the public LB (target 1200+ Elo; current #1 "flg" = 1210, 5792 teams).
Deadline 2026-08-16, then ~2 weeks of continued play. User gives full token budget
for ~2 days (day 1 was 2026-07-27). User uploads submissions manually; Kaggle CLI
is authenticated and user has entered the competition.

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

## Experimental results so far (local arena, seat-swapped)

vs tuned sample rule agents (rule:iono ≈ LB Elo 800):
- search + handcrafted eval only: **50%** vs rule:iono (mirror), 33% vs rule:dragapult.
- **3× compute made it WORSE (17%)** → handcrafted eval misleads deeper search.
- value-net leaf (v1, 768 games, overfit): **0%** — worse than handcrafted.
- BC-only (policy v1, 57% top-1 val): **10%** vs rule:iono (loses the prize race;
  plays coherently — benches, Rare-Candy evolves, takes prizes, just too leaky).
- **Hybrid (policy playouts + handcrafted leaf, SA_NO_VNET=1): 50% vs rule:iono
  cross-deck (grimmsnarl vs iono)** — best so far, wiring works.

Bottleneck identified: **net quality, driven by data volume** (was 768 games).
Late-game (turn≥14) win prediction is genuinely hard (~50% even for logreg probe);
mid-game ~71% (logreg baseline). Value net must beat logreg (0.634 overall) to be useful.

## Data pipeline

- `scripts/fetch_top_episodes.py --date YYYY-MM-DD --max N` — downloads manifest.csv
  of daily dataset `kaggle/pokemon-tcg-ai-battle-episodes-<date>`, then top-N episodes
  by avg_score into `replays/<date>/`. Idempotent (skips existing). Now parallel (4 workers).
- Old repo has 366 top-1% replays: `E:\Kaggle\pokemon-tcg-simulation\replay_miner\replays\2026-07-06..12`.
- Downloaded so far: `replays/2026-07-26` (403), `2026-07-25` (~230+, in flight),
  `2026-07-23` (~140+, in flight), 07-24 queued after 25, then 22/21.
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

- `python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent search`
  → builds + **smoke-tests** `dist/submission_*.tar.gz` (self-play with time pool,
  asserts pool not exhausted). `--agent bc` for policy-only bundle.
- NOT yet submitted anything. First submission should go up ASAP once hybrid+v2 nets
  beat rule agents convincingly (validates Linux env + starts rating grind).
  User must upload manually (or `kaggle competitions submit -f dist/submission.tar.gz -m "msg"`).

## Immediate next steps (priority order)

1. Wait/re-run v2 dataset rebuild + retrain (was in flight). Confirm policy top-1 ≥ 60%
   and value val_loss < logreg baseline. More data days keep arriving — retrain again after.
2. Arena matrix: {bc, hybrid, hybrid-noV} × {rule:iono, rule:dragapult} + mirrors
   (grimmsnarl vs crispin_box). Need >75% vs rules before submitting.
3. Build submission (grimmsnarl + best agent config), smoke, give to user for upload.
4. Iterate day 2: more data (07-24/23/22/21 + deepen to top-800/day), winners-only
   policy variant, more epochs, maybe playout temperature. Watch for causal-confusion
   ceiling of BC (paper in papers/ warns); search-on-top is our hedge.
5. Deck A/B: grimmsnarl vs crispin_box with the SAME agent; also consider flg's
   secondary P/R deck (55 games in 07-26 data).

## Gotchas

- Windows: use `python -X utf8` everywhere (cp1252 crashes on card names).
- Arena/harness must run from repo root; `sys.path` needs `src/`, `agents/`, root.
- Background bash jobs die with the session; re-launch fetches idempotently.
- Old repo `E:\Kaggle\pokemon-tcg-simulation` = failed pure-RL attempts; don't import
  its approach, only its replays. gitignore keeps data/replays/artifacts/dist out of git.
- Commit style: fine-grained, one-line semantic messages + Claude co-author trailer.
