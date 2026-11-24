"""Turn teacher games into compact Transformer distillation shards.

Input: the `.jsonl.gz` files written by `scripts/teacher_games.py`.
Output: `{split}-{input}-{k:03d}.npz` shards plus `manifest.json`, read by
`agents/tx/train.py`.

What each teacher decision becomes. A decision with action [a1..ak] expands
into the upstream sequence of pointer queries: pick a1, then a2 given a1, ...,
then STOP if k < maxCount (ptcg-population-rl `action_queries`). Each query
stores:

  - its tokens, SPARSE: (position, value) pairs with position = row*198 + col.
    Tokens are ~94% zeros; dense float32 for 30k games is ~72 GB, this is ~6 GB.
  - the teacher's score for every candidate token. Options get the net's
    logit. STOP gets a synthetic score halfway between the lowest picked and
    the highest unpicked logit, which makes "take these k, then stop" the
    teacher's argmax at every step. (The trainer only applies the teacher's
    distribution on a decision's FIRST query; later queries train on the hard
    target alone.)
  - the target token (the teacher's pick, or STOP).

Per decision: number of queries, acting seat, and terminal outcome from that
seat (+1 win, -1 loss, 0 draw) for the value head.

Split: whole games, by a hash of the game id (5% validation). Not `id % 20`:
the teacher mix runs a 32-game cycle, and a modulus would skip whole opponents.

    python -X utf8 scripts/tx_extract.py --input out/teacher --output out/tx_data
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from tx.deck import build, engine_tables  # noqa: E402
from tx.features import WIDTH, Encoder  # noqa: E402

VALIDATION_EVERY = 20          # 1 in 20 games, by hash
GAMES_PER_SHARD = 1000


def split_of(gid: int) -> str:
    h = int(hashlib.sha256(str(gid).encode()).hexdigest()[:8], 16)
    return "validation" if h % VALIDATION_EVERY == 0 else "train"


def stop_score(scores: list[float], action: list[int]) -> float:
    """A STOP logit that ranks below every pick and above every non-pick."""
    picked = set(action)
    lo = min((scores[i] for i in picked), default=None)
    hi = max((s for i, s in enumerate(scores) if i not in picked), default=None)
    if lo is None and hi is None:
        return 0.0
    if lo is None:
        return hi + 1.0
    if hi is None:
        return lo - 1.0
    return (lo + hi) / 2.0


class Buffer:
    """Accumulates one split's queries and writes a shard when full."""

    def __init__(self, out: Path, split: str, stem: str):
        self.out, self.split, self.stem = out, split, stem
        self.k = 0
        self.shards: list[dict] = []
        self.reset()

    def reset(self):
        self.pos, self.val, self.nnz, self.size = [], [], [], []
        self.target, self.cand_idx, self.cand_score, self.n_cand = [], [], [], []
        self.d_size, self.d_outcome, self.d_game, self.d_seat, self.d_index = [], [], [], [], []
        self.games: list[int] = []

    def add_query(self, tokens: np.ndarray, cand: list[int], score: list[float], target: int):
        flat = tokens.reshape(-1)
        nz = np.flatnonzero(flat)
        self.pos.append(nz.astype(np.uint16))
        self.val.append(flat[nz])
        self.nnz.append(len(nz))
        self.size.append(len(tokens))
        self.target.append(target)
        self.cand_idx.append(np.asarray(cand, np.uint8))
        self.cand_score.append(np.asarray(score, np.float32))
        self.n_cand.append(len(cand))

    def flush(self):
        if not self.games:
            return
        name = f"{self.split}-{self.stem}-{self.k:03d}.npz"
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
        arrays = dict(
            pos=cat(self.pos, np.uint16), val=cat(self.val, np.float32),
            nnz=np.asarray(self.nnz, np.int32), size=np.asarray(self.size, np.uint8),
            target=np.asarray(self.target, np.uint8),
            cand_idx=cat(self.cand_idx, np.uint8), cand_score=cat(self.cand_score, np.float32),
            n_cand=np.asarray(self.n_cand, np.uint8),
            d_size=np.asarray(self.d_size, np.int16), d_outcome=np.asarray(self.d_outcome, np.int8),
            d_game=np.asarray(self.d_game, np.int64), d_seat=np.asarray(self.d_seat, np.int8),
            d_index=np.asarray(self.d_index, np.int32))
        np.savez_compressed(self.out / name, **arrays)
        digest = hashlib.sha256((self.out / name).read_bytes()).hexdigest()
        self.shards.append(dict(path=name, sha256=digest, games=len(self.games),
                                decisions=len(self.d_size), queries=len(self.size),
                                nnz=int(sum(self.nnz))))
        self.k += 1
        self.reset()


def extract_file(job: tuple[str, str, str]) -> dict:
    """Extract one teacher `.jsonl.gz` into train/validation shards."""
    src, out, profile_name = job
    src, out = Path(src), Path(out)
    cards, attacks = engine_tables()
    from arena import resolve_deck

    canonical, deck = resolve_deck(profile_name)
    profile = build(canonical, list(deck), cards, attacks)
    enc = Encoder(profile, cards)
    stem = src.name.split(".")[0]
    bufs = {s: Buffer(out, s, stem) for s in ("train", "validation")}
    games: dict[str, list[dict]] = {"train": [], "validation": []}
    stats = dict(games=0, decisions=0, queries=0, skipped_unscored=0, skipped_disagree=0)
    models: set[str] = set()
    t0 = time.time()

    header = None
    pending: list[dict] = []
    with gzip.open(src, "rt") as f:
        for line in f:
            r = json.loads(line)
            kind = r["type"]
            if kind == "header":
                header, pending = r, []
                for s in r["teacher_seats"]:
                    profile.check_deck(r["decks"][s])
                    models.add(r["models"][s])
                continue
            if kind == "decision":
                pending.append(r)
                continue
            # terminal: the game is complete, so its decisions can be written
            gid, result = r["game"], r["result"]
            split = split_of(gid)
            b = bufs[split]
            for d in pending:
                scores, action, obs = d["teacher_scores"], d["action"], d["observation"]
                if scores is None:
                    stats["skipped_unscored"] += 1
                    continue
                if d["net_agrees"] is False:
                    stats["skipped_disagree"] += 1
                    continue
                sel = obs["select"]
                # Picks in teacher-score order. The net's own picks already are;
                # a forced take-everything select returns index order, and the
                # pointer should learn the teacher's ranking, not index order.
                # Stable, so tied picks keep the agent's order.
                action = sorted(action, key=lambda i: -scores[i])
                s_stop = stop_score(scores, action)
                base = enc.tokenize(obs)
                prefix: list[int] = []
                nq = 0
                for target in [*action, *([-1] if len(action) < sel["maxCount"] else [])]:
                    options = enc.options_for(obs, prefix)
                    if target not in options:
                        raise ValueError(f"game {gid}: action {action} is not a legal sequence")
                    toks = enc.with_prefix(base, obs, prefix)
                    idx = enc.token_indices(obs, options)
                    sc = [scores[i] if i >= 0 else s_stop for i in options]
                    b.add_query(toks, idx, sc, idx[options.index(target)])
                    nq += 1
                    if target != -1:
                        prefix.append(target)
                seat = d["seat"]
                b.d_size.append(nq)
                b.d_outcome.append(0 if result == 2 else (1 if result == seat else -1))
                b.d_game.append(gid)
                b.d_seat.append(seat)
                b.d_index.append(d["index"])
                stats["decisions"] += 1
                stats["queries"] += nq
            b.games.append(gid)
            games[split].append(dict(game=gid, result=result, agents=header["agents"],
                                     teacher_seats=header["teacher_seats"]))
            stats["games"] += 1
            if len(b.games) >= GAMES_PER_SHARD:
                b.flush()
            if stats["games"] % 1000 == 0:
                print(f"{src.name}: {stats['games']} games "
                      f"{stats['games'] / (time.time() - t0):.1f} games/s", flush=True)
    for b in bufs.values():
        b.flush()
    return dict(source=src.name, stats=stats, models=sorted(models),
                shards={s: b.shards for s, b in bufs.items()}, games=games,
                profile=profile.to_json(), profile_id=profile.profile_id,
                contract=enc.contract, contract_id=enc.contract_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="dir of teacher .jsonl.gz files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--deck", default="grimmsnarl", help="the teacher's deck module")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    src = sorted(Path(args.input).rglob("*.jsonl.gz"))
    if not src:
        raise SystemExit(f"no .jsonl.gz under {args.input}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    t0 = time.time()
    with Pool(min(args.workers, len(src))) as pool:
        parts = pool.map(extract_file, [(str(s), str(out), args.deck) for s in src])
    ids = {p["contract_id"] for p in parts}
    models = {m for p in parts for m in p["models"]}
    if len(ids) != 1 or len(models) != 1:
        raise SystemExit(f"mixed contracts {ids} or teacher models {models}")
    splits = {s: dict(shards=[sh for p in parts for sh in p["shards"][s]],
                      games=[g for p in parts for g in p["games"][s]])
              for s in ("train", "validation")}
    seen = [g["game"] for s in splits.values() for g in s["games"]]
    if len(seen) != len(set(seen)):
        raise SystemExit("a game id appears twice")
    totals = {k: sum(p["stats"][k] for p in parts) for k in parts[0]["stats"]}
    manifest = dict(format="scio-tx-distill-v1", width=WIDTH,
                    contract=parts[0]["contract"], contract_id=parts[0]["contract_id"],
                    profile=parts[0]["profile"], profile_id=parts[0]["profile_id"],
                    teacher_model_md5=models.pop(), sources=[p["source"] for p in parts],
                    totals=totals, splits=splits)
    (out / "manifest.json").write_text(json.dumps(manifest) + "\n")
    mb = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(json.dumps(dict(totals=totals,
                          train_games=len(splits["train"]["games"]),
                          validation_games=len(splits["validation"]["games"]),
                          shards=sum(len(s["shards"]) for s in splits.values()),
                          output_mb=round(mb, 1), minutes=round((time.time() - t0) / 60, 1))),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
