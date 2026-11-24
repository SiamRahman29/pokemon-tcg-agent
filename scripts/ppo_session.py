"""One PPO session: run cycles until the time budget, then save state to resume.

A cycle: freeze the current policy as `.npz`; play `--games` sampled games
across `--workers` processes against the weighted opponent pool, recording
the learner's decisions; run one clipped-PPO pass (agents/tx/ppo.py); save.

State lives in `<output>/state/` and is complete after every cycle:
  ppo_state.json   cycle, learning rate, opponent pool and recent results
  model.pt         weights + optimizer state
  snapshots/*.npz  frozen opponents (snapshot 0 is the distilled model)
  history.jsonl    one line per cycle; evaluations too
  candidates/      checkpoints whose evaluation beat the distilled model

Start a chain from a distilled checkpoint (`--init`), or resume from the
newest `ppo_state.json` found under `--resume` (Kaggle: /kaggle/input, the
previous session's output). On a fresh start the learning rate is first
calibrated toward the held-out KL target, as upstream does.

Opponent weighting (frozen 2026-11-24): `self` gets a fixed 20% of games;
every other opponent gets weight 0.2 + (1 - recent score), recent = mean over
its last 3 cycles (0.5 when unseen). A frozen snapshot joins every
`--snapshot-every` cycles. Not upstream: if the mean held-out KL of the last
3 cycles leaves [0.5x, 2x] the target, the LR is rescaled toward it (bounded
0.5x-2x per step), since a long chain cannot be recalibrated by hand.

    python -X utf8 scripts/ppo_session.py --init out/tx_model/model.pt \\
        --output out/ppo0 --max-hours 11
    python -X utf8 scripts/ppo_session.py --resume /kaggle/input --output out/ppo0
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

SELF_SHARE = 0.20
RECENT = 3
DECK = "grimmsnarl"
TEACHER = "bc:v5s2,net=out/policy_v5_s2.npz,noChip,noSpread,noSrc"
BASE_POOL = [
    dict(name="v5s2", kind="arena", spec=TEACHER, deck="grimmsnarl"),
    dict(name="v10", kind="arena", spec="rule:v10,noS", deck="lucario_v10"),
    dict(name="crustle", kind="arena", spec="rule:crustle", deck="crustle_v1"),
    dict(name="alakazam", kind="arena", spec="rule:alakazam5", deck="alakazam5"),
    dict(name="archaludon", kind="arena", spec="rule:archaludon", deck="archaludon_ex"),
    dict(name="dragapult", kind="arena", spec="rule:dragapult", deck="dragapult_ex"),
    dict(name="iono", kind="arena", spec="rule:iono", deck="iono"),
    dict(name="abomasnow", kind="arena", spec="rule:abomasnow", deck="mega_abomasnow_ex"),
    dict(name="lucario", kind="arena", spec="rule:lucario", deck="mega_lucario_ex"),
]


# ---- rollout worker (separate process) --------------------------------------

def collect_part(job: dict) -> dict:
    """Play this worker's games; write packed rollout arrays to `job['out']`."""
    from ptcg.env import harness, sdk

    sdk.load()
    from arena import build_agent, resolve_deck
    from tx.deck import engine_tables, for_deck_module
    from tx.features import Encoder
    from tx.numpy_policy import NumpyTransformer
    from tx.ppo import Sampler, pack_episodes

    cards, _ = engine_tables()
    enc = Encoder(for_deck_module(DECK), cards)
    _, my_deck = resolve_deck(DECK)
    learner = NumpyTransformer(job["model"], enc)
    rng = np.random.default_rng(job["seed"])
    frozen: dict[str, NumpyTransformer] = {}
    pool = {o["name"]: o for o in job["pool"]}
    episodes: list[dict] = []
    results: dict[str, list[float]] = {}
    errors = 0
    for g, (name, seat) in enumerate(job["games"]):
        opp = pool[name]
        me = Sampler(learner, enc, my_deck, rng, job["temperature"])
        if name == "self":
            them, opp_deck = Sampler(learner, enc, my_deck, rng, job["temperature"]), list(my_deck)
        elif opp["kind"] == "snapshot":
            if name not in frozen:
                frozen[name] = NumpyTransformer(job["snapshot_dir"] + "/" + opp["file"], enc)
            them = Sampler(frozen[name], enc, my_deck, rng, job["temperature"], record=False)
            opp_deck = list(my_deck)
        else:
            deck_name, opp_deck = resolve_deck(opp["deck"])
            _, them = build_agent(opp["spec"], list(opp_deck), deck_name)
        a0, a1 = (me, them) if seat == 0 else (them, me)
        d0, d1 = (list(my_deck), list(opp_deck)) if seat == 0 else (list(opp_deck), list(my_deck))
        try:
            r = harness.play_game(a0, a1, d0, d1)
        except Exception as e:  # noqa: BLE001 -- one failed game must not sink the cycle
            errors += 1
            print(f"worker {job['worker']} game vs {name}: {type(e).__name__}: {e}", flush=True)
            continue
        learners = [(seat, me)] + ([(1 - seat, them)] if name == "self" else [])
        for s, agent in learners:
            if agent.decisions:
                reward = 0.0 if r.winner == 2 else (1.0 if r.winner == s else -1.0)
                episodes.append(dict(episode=job["episode_base"] + 2 * g + s, reward=reward,
                                     decisions=agent.decisions))
        if name != "self":
            results.setdefault(name, []).append(0.5 if r.winner == 2 else float(r.winner == seat))
    packed = pack_episodes(episodes)
    np.savez(job["out"], **packed)
    return dict(worker=job["worker"], games=len(job["games"]) - errors, errors=errors,
                episodes=len(episodes), decisions=int(len(packed["old_logp"])), results=results)


# ---- session ----------------------------------------------------------------

def allocate(state: dict, games: int) -> dict[str, int]:
    """Games per opponent: fixed self share, the rest weighted toward losses."""
    others = [o for o in state["pool"] if o["name"] != "self"]
    weights = []
    for o in others:
        recent = state["recent"].get(o["name"], [])[-RECENT:]
        score = float(np.mean(recent)) if recent else 0.5
        weights.append(0.2 + (1.0 - score))
    weights = np.asarray(weights) / sum(weights)
    rest = games - int(round(games * SELF_SHARE))
    raw = weights * rest
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: rest - counts.sum()]:
        counts[i] += 1
    alloc = {o["name"]: int(c) for o, c in zip(others, counts)}
    alloc["self"] = games - rest
    return alloc


def collect(state: dict, model_npz: Path, sdir: Path, games: int, workers: int,
            temperature: float, tmp: Path) -> tuple[dict, list[dict]]:
    from tx.ppo import merge

    alloc = allocate(state, games)
    plan = [(name, (i % 2)) for name, n in alloc.items() for i in range(n)]
    rng = np.random.default_rng(1000 + state["cycle"])
    rng.shuffle(plan)
    jobs = []
    for w in range(workers):
        jobs.append(dict(worker=w, games=plan[w::workers], model=str(model_npz),
                         pool=state["pool"], snapshot_dir=str(sdir), temperature=temperature,
                         seed=int(rng.integers(2**31)), out=str(tmp / f"part{w}.npz"),
                         episode_base=state["cycle"] * 10_000_000 + w * 1_000_000))
    with get_context("spawn").Pool(workers) as pool:
        parts = pool.map(collect_part, jobs)
    data = merge([dict(np.load(j["out"])) for j in jobs])
    for j in jobs:
        Path(j["out"]).unlink()
    return data, parts


def load_model(path: Path):
    import torch

    from tx.model import ARCHITECTURE, ObservationTransformer

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["architecture"] != ARCHITECTURE:
        raise SystemExit("architecture mismatch")
    model = ObservationTransformer(payload["contract"]["width"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def save_state(out: Path, state: dict, model, optimizer, payload: dict) -> None:
    import torch

    sd = out / "state"
    tmp = sd / "model.pt.tmp"
    torch.save(dict({k: payload[k] for k in ("contract", "contract_id", "architecture", "profile",
                                             "profile_id", "teacher_model_md5")},
                    format="scio-tx-checkpoint-v1", stage="ppo", cycle=state["cycle"],
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    optimizer_state=optimizer.state_dict()), tmp)
    tmp.replace(sd / "model.pt")
    (sd / "ppo_state.json").write_text(json.dumps(state, indent=1) + "\n")


def find_state(root: Path) -> Path | None:
    hits = sorted(root.rglob("ppo_state.json"),
                  key=lambda p: json.loads(p.read_text())["cycle"])
    return hits[-1].parent if hits else None


def find_checkpoint(root: Path) -> Path:
    import torch

    for p in sorted(root.rglob("model.pt")):
        if torch.load(p, map_location="cpu", weights_only=True).get("format") == "scio-tx-checkpoint-v1":
            return p
    raise SystemExit(f"no distilled checkpoint under {root}")


def evaluate(out: Path, model_npz: Path, distilled_npz: Path, cycle: int, workers: int,
             scale: float = 1.0) -> dict:
    """Greedy evaluation in `workers` parallel shards via scripts/tx_eval.py."""
    edir = out / "evals" / f"cycle{cycle:04d}"
    procs = [subprocess.Popen(
        [sys.executable, "-X", "utf8", str(ROOT / "scripts/tx_eval.py"), "--model", str(model_npz),
         "--distilled", str(distilled_npz), "--shard", str(i), "--shards", str(workers),
         "--scale", str(scale),
         "--out", str(edir / f"s{i}")], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        for i in range(workers)]
    for p in procs:
        p.wait()
    r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/tx_eval.py"),
                        "--summarize", str(edir)], capture_output=True, text=True)
    print(r.stdout, flush=True)
    rows = json.loads((edir / "summary.json").read_text())
    return {row["opponent"]: row for row in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, help="distilled model.pt, or a dir to search")
    ap.add_argument("--resume", type=Path, help="dir to search for a previous session's state")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--max-cycles", type=int, default=10_000)
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="torch threads for the update")
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-scale", type=float, default=1.0,
                    help="scale evaluation match counts (smoke tests)")
    ap.add_argument("--rates", type=float, nargs="+", default=[1.5e-5, 3e-5, 6e-5, 1.2e-4],
                    help="learning rates tried by the first-cycle calibration")
    ap.add_argument("--updates", type=int, help="override PPO updates per cycle (smoke tests)")
    ap.add_argument("--batch-size", type=int, help="override PPO batch size (smoke tests)")
    a = ap.parse_args()

    import torch

    from tx.ppo import DEFAULTS, audit_behavior, export_npz, update

    t0 = time.time()
    out = a.output
    sd = out / "state"
    snap_dir = sd / "snapshots"
    prev = find_state(a.resume) if a.resume and a.resume.exists() else None
    if prev is not None:
        shutil.copytree(prev, sd, dirs_exist_ok=True)
        state = json.loads((sd / "ppo_state.json").read_text())
        model, payload = load_model(sd / "model.pt")
        print(f"resumed at cycle {state['cycle']} from {prev}", flush=True)
    else:
        if a.init is None:
            raise SystemExit("no previous state found; pass --init")
        init = a.init if a.init.suffix == ".pt" else find_checkpoint(a.init)
        model, payload = load_model(init)
        snap_dir.mkdir(parents=True, exist_ok=True)
        export_npz(model, payload, snap_dir / "snap0000.npz")
        state = dict(cycle=0, learning_rate=None, calibrated=False, recent={},
                     pool=[dict(name="self", kind="self")] + BASE_POOL
                     + [dict(name="snap0000", kind="snapshot", file="snap0000.npz")],
                     init_checkpoint=str(init))
        (sd / "history.jsonl").touch()
        print(f"fresh start from {init}", flush=True)
    config = dict(DEFAULTS)
    if a.updates:
        config["updates"] = a.updates
    if a.batch_size:
        config["batch_size"] = a.batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    if "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    live = tmp / "live.npz"
    history = (sd / "history.jsonl").open("a")
    last_cycle_s = None
    done_here = 0

    while done_here < a.max_cycles:
        elapsed = time.time() - t0
        need = (last_cycle_s or 3600) * 1.25
        if elapsed + need > a.max_hours * 3600:
            print(f"stopping: {elapsed / 3600:.2f} h used, next cycle needs ~{need / 3600:.2f} h",
                  flush=True)
            break
        c0 = time.time()
        export_npz(model, dict(payload, stage="ppo", cycle=state["cycle"]), live)
        data, parts = collect(state, live, snap_dir, a.games, a.workers,
                              config["temperature"], tmp)
        t_collect = time.time() - c0
        audit = audit_behavior(model, data, config["temperature"], sample=2000,
                               seed=state["cycle"])
        if max(audit["max_abs_errors"].values()) > 1e-3:
            raise SystemExit(f"behaviour audit failed: {audit}")
        if not state["calibrated"]:
            trials = []
            for rate in a.rates:
                m = copy.deepcopy(model)
                o = torch.optim.AdamW(m.parameters(), lr=rate, weight_decay=config["weight_decay"])
                if "optimizer_state" in payload:
                    o.load_state_dict(optimizer.state_dict())
                rep = update(m, o, data, dict(config, learning_rate=rate), a.threads)
                trials.append(dict(learning_rate=rate, kl=rep["heldout_global_approximate_kl"]))
                print(json.dumps(dict(calibration=trials[-1])), flush=True)
            best = min(trials, key=lambda t: abs(t["kl"] - config["kl_target"]))
            state.update(learning_rate=best["learning_rate"], calibrated=True)
            history.write(json.dumps(dict(cycle=state["cycle"], calibration=trials,
                                          selected=best)) + "\n")
            history.flush()
            # Upstream: trial weights are never deployed; the next cycle
            # collects a fresh rollout at the selected rate.
            save_state(out, state, model, optimizer, payload)
            last_cycle_s = time.time() - c0
            continue
        rep = update(model, optimizer, data, dict(config, learning_rate=state["learning_rate"]),
                     a.threads)
        state["cycle"] += 1
        for part in parts:
            for name, rs in part["results"].items():
                state["recent"].setdefault(name, []).append(float(np.mean(rs)))
                state["recent"][name] = state["recent"][name][-RECENT:]
        kls = [rep["heldout_global_approximate_kl"]] + state.get("recent_kl", [])
        state["recent_kl"] = kls[:3]
        if len(state["recent_kl"]) == 3:
            mean_kl = float(np.mean(state["recent_kl"]))
            if not 0.5 * config["kl_target"] <= mean_kl <= 2 * config["kl_target"]:
                factor = float(np.clip(config["kl_target"] / max(mean_kl, 1e-9), 0.5, 2.0))
                state["learning_rate"] *= factor
                state["recent_kl"] = []
                print(f"LR rescaled x{factor:.2f} -> {state['learning_rate']:.2e}", flush=True)
        line = dict(cycle=state["cycle"], seconds_collect=round(t_collect),
                    seconds_total=round(time.time() - c0),
                    games=sum(p["games"] for p in parts), errors=sum(p["errors"] for p in parts),
                    decisions=int(len(data["old_logp"])), audit=audit["max_abs_errors"],
                    scores={n: round(float(np.mean(v)), 3) for n, v in state["recent"].items()},
                    allocation=allocate(state, a.games), **{k: rep[k] for k in (
                        "heldout_global_approximate_kl", "learning_rate", "parameter_delta_l2",
                        "mean_metrics")})
        del data
        if state["cycle"] % a.snapshot_every == 0:
            name = f"snap{state['cycle']:04d}"
            export_npz(model, dict(payload, stage="ppo", cycle=state["cycle"]),
                       snap_dir / f"{name}.npz")
            state["pool"].append(dict(name=name, kind="snapshot", file=f"{name}.npz"))
            line["snapshot"] = name
        save_state(out, state, model, optimizer, payload)
        if state["cycle"] % a.eval_every == 0:
            export_npz(model, dict(payload, stage="ppo", cycle=state["cycle"]), live)
            ev = evaluate(out, live, snap_dir / "snap0000.npz", state["cycle"], a.workers,
                          a.eval_scale)
            line["eval"] = {k: dict(score=round(v["score"], 3), lo=round(v["lo"], 3),
                                    games=v["games"]) for k, v in ev.items()}
            d = ev.get("distilled")
            if d and d["lo"] > 0.5:
                cand = sd / "candidates"
                cand.mkdir(exist_ok=True)
                shutil.copy(live, cand / f"cycle{state['cycle']:04d}.npz")
                line["candidate"] = True
        history.write(json.dumps(line) + "\n")
        history.flush()
        print(json.dumps({k: v for k, v in line.items() if k not in ("allocation", "mean_metrics")}),
              flush=True)
        last_cycle_s = time.time() - c0
        done_here += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"session done: cycle {state['cycle']}, {done_here} cycles this session, "
          f"{(time.time() - t0) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
