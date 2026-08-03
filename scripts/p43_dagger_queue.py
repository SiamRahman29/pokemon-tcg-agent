"""E3: curate uncertain decisions from the frozen clone's live trajectories.

The fresh rating-977 replays were produced by the decision-identical v5 policy,
so they are the strongest available DAgger source: states the shipped clone
actually visits against the live field.  This script ranks only faithful,
non-forced decisions by the clone's selected/unselected logit boundary.

Outcomes are deliberately omitted from the queue.  A reviewer must label the
state in front of them, not learn which move happened to precede a win.

    python -X utf8 scripts/p43_dagger_queue.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa import cards, policynet  # noqa: E402
from sa.optfeat import option_features  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def selected(v: dict, n_options: int) -> list[int] | None:
    action = v.get("selected")
    if action is None:
        action = v.get("action")
    if not isinstance(action, list):
        return None
    picked = [int(x) for x in action
              if isinstance(x, int) and 0 <= x < n_options]
    return picked if len(picked) == len(action) else None


def softmax_stats(scores: np.ndarray) -> tuple[list[float], float]:
    z = scores.astype(np.float64) - float(scores.max())
    probs = np.exp(z)
    probs /= probs.sum()
    if len(probs) <= 1:
        entropy = 0.0
    else:
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        entropy /= math.log(len(probs))
    return [float(x) for x in probs], entropy


def option_signature(obs: dict, option: dict) -> bytes:
    dense, card_id, attack_id, target_id = option_features(obs, option)
    return (np.asarray(dense, dtype=np.float32).tobytes()
            + np.asarray([card_id, attack_id, target_id],
                         dtype=np.int32).tobytes())


def card_name(card_id: int) -> str:
    if not card_id:
        return ""
    return str(cards.card(int(card_id)).get("name") or f"card {card_id}")


def attack_name(attack_id: int) -> str:
    if not attack_id:
        return ""
    return str(cards.attacks().get(int(attack_id), {}).get("name")
               or f"attack {attack_id}")


def option_view(obs: dict, option: dict, index: int, score: float,
                probability: float, clone_selected: bool) -> dict[str, Any]:
    _, card_id, attack_id, target_id = option_features(obs, option)
    labels = []
    if card_id:
        labels.append(card_name(card_id))
    if attack_id:
        labels.append(attack_name(attack_id))
    if target_id:
        labels.append(f"target {card_name(target_id)}")
    if not labels:
        labels.append(
            ", ".join(f"{k}={v}" for k, v in sorted(option.items()))
            or f"option {index}")
    return {
        "index": index,
        "label": " · ".join(labels),
        "card_id": int(card_id),
        "attack_id": int(attack_id),
        "target_id": int(target_id),
        "score": float(score),
        "probability": probability,
        "clone_selected": clone_selected,
        "raw": option,
    }


def pokemon_view(pokemon: Any) -> dict[str, Any] | None:
    if not pokemon:
        return None
    if isinstance(pokemon, list):
        pokemon = pokemon[0] if pokemon else None
    if not isinstance(pokemon, dict):
        return None
    cid = int(pokemon.get("id") or 0)
    return {
        "name": card_name(cid),
        "card_id": cid,
        "hp": pokemon.get("hp"),
        "max_hp": pokemon.get("maxHp"),
        "energies": list(pokemon.get("energies") or []),
        "tools": [card_name(int(x.get("id") or 0))
                  for x in (pokemon.get("tools") or []) if isinstance(x, dict)],
    }


def player_view(player: dict) -> dict[str, Any]:
    hand = player.get("hand")
    return {
        "active": pokemon_view(player.get("active")),
        "bench": [x for x in
                  (pokemon_view(p) for p in (player.get("bench") or [])) if x],
        "hand": ([card_name(int(c.get("id") or 0)) for c in hand
                  if isinstance(c, dict)] if isinstance(hand, list) else None),
        "hand_count": int(player.get("handCount") or
                          (len(hand) if isinstance(hand, list) else 0)),
        "deck_count": int(player.get("deckCount") or 0),
        "prizes_remaining": len(player.get("prize") or []),
        "discard": [card_name(int(c.get("id") or 0))
                    for c in (player.get("discard") or [])
                    if isinstance(c, dict)],
    }


def board_view(obs: dict) -> dict[str, Any]:
    state = obs["current"]
    me = int(state["yourIndex"])
    players = state.get("players") or [{}, {}]
    stadium = state.get("stadium") or []
    stadium_card = stadium[0] if stadium and isinstance(stadium[0], dict) else {}
    return {
        "turn": state.get("turn"),
        "turn_action_count": state.get("turnActionCount"),
        "acting_seat": me,
        "first_player": state.get("firstPlayer"),
        "stadium": card_name(int(stadium_card.get("id") or 0)),
        "you": player_view(players[me]),
        "opponent": player_view(players[1 - me]),
    }


def make_candidate(path: Path, replay: dict, visual_index: int, v: dict,
                   player: str, net: policynet.Net) -> tuple[dict | None, bool]:
    obs = v.get("obs") or {}
    state = obs.get("current") or {}
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    if state.get("result") != -1 or len(options) < 2:
        return None, False
    names = (replay.get("info") or {}).get("TeamNames") or []
    me = int(state.get("yourIndex", -1))
    if me not in (0, 1) or me >= len(names) or names[me] != player:
        return None, False
    actual = selected(v, len(options))
    if actual is None:
        return None, False
    mn = int(sel.get("minCount") or 0)
    mx = int(sel.get("maxCount") or 0)
    if mx == 0 or (mn == mx == len(options)):
        return None, False

    scores = net.scores(obs)
    clone = net.choose(obs)
    faithful = set(actual) == set(clone)
    if not faithful:
        return None, True
    chosen = set(clone)
    unchosen = set(range(len(options))) - chosen
    # The first E3 round targets ranking errors. If every option is selected,
    # the missing signal is a stop/count logit, which v5 does not have.
    if not chosen or not unchosen:
        return None, True
    low_chosen = min(chosen, key=lambda i: float(scores[i]))
    high_unchosen = max(unchosen, key=lambda i: float(scores[i]))
    if option_signature(obs, options[low_chosen]) == option_signature(
            obs, options[high_unchosen]):
        # Two copies of the same card in the same role are a free tie (§8x).
        return None, True
    margin = float(scores[low_chosen] - scores[high_unchosen])
    probs, entropy = softmax_stats(scores)
    replay_id = str(replay.get("id") or path.stem)
    item_id = hashlib.sha1(
        f"{replay_id}:{visual_index}:{me}".encode("utf-8")).hexdigest()[:16]
    item = {
        "id": item_id,
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "replay_id": replay_id,
        "visual_index": visual_index,
        "acting_seat": me,
        "select_type": int(sel.get("type") or 0),
        "context": int(sel.get("context") or 0),
        "min_count": mn,
        "max_count": mx,
        "n_options": len(options),
        "boundary_margin": margin,
        "normalized_entropy": entropy,
        "clone_action": sorted(int(x) for x in clone),
        "board": board_view(obs),
        "select": {
            k: value for k, value in sel.items() if k != "option"
        },
        "options": [
            option_view(obs, option, i, float(scores[i]), probs[i], i in chosen)
            for i, option in enumerate(options)
        ],
        # Kept for corpus export after review. The UI does not expose outcome
        # because hindsight would contaminate the teacher label.
        "observation": obs,
    }
    return item, True


def curate(candidates: list[dict], size: int, per_replay: int,
           per_bucket: int) -> list[dict]:
    ranked = sorted(
        candidates,
        key=lambda x: (x["boundary_margin"], -x["normalized_entropy"], x["id"]))
    out: list[dict] = []
    seen: set[str] = set()
    replay_counts: Counter = Counter()
    bucket_counts: Counter = Counter()

    def take(item: dict, enforce_bucket: bool) -> bool:
        rid = item["replay_id"]
        bucket = (item["select_type"], item["context"])
        if replay_counts[rid] >= per_replay:
            return False
        if enforce_bucket and bucket_counts[bucket] >= per_bucket:
            return False
        out.append(item)
        seen.add(item["id"])
        replay_counts[rid] += 1
        bucket_counts[bucket] += 1
        return True

    for item in ranked:
        if len(out) >= size:
            break
        take(item, True)
    # Fill a short queue by relaxing context quotas, never the replay cap.
    for item in ranked:
        if len(out) >= size:
            break
        if item["id"] not in seen:
            take(item, False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays",
                    default="replays/submission_v5_003/submission_977")
    ap.add_argument("--player", default="Scio")
    ap.add_argument("--net", default="out/policy_v5.npz")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--per-replay", type=int, default=3)
    ap.add_argument("--per-bucket", type=int, default=24,
                    help="first-pass cap per (select type, context)")
    ap.add_argument("--out", default="out/e3/review_queue.jsonl")
    args = ap.parse_args()

    net_path = ROOT / args.net
    net = policynet.load(net_path)
    if net is None:
        raise SystemExit(f"{net_path} did not load")
    paths = sorted((ROOT / args.replays).glob("*.json"))
    if not paths:
        raise SystemExit(f"no replay JSON files under {ROOT / args.replays}")

    candidates: list[dict] = []
    eligible = mismatched = errors = 0
    for path in paths:
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
            vis = replay["steps"][0][0].get("visualize") or []
            for i, v in enumerate(vis):
                item, considered = make_candidate(
                    path, replay, i, v, args.player, net)
                if considered:
                    eligible += 1
                    if item is None:
                        obs = v.get("obs") or {}
                        sel = obs.get("select") or {}
                        actual = selected(v, len(sel.get("option") or []))
                        if actual is not None and obs.get("current"):
                            try:
                                clone = net.choose(obs)
                                mismatched += set(actual) != set(clone)
                            except Exception:
                                pass
                if item is not None:
                    candidates.append(item)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if errors <= 5:
                print(f"{path.name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)

    fidelity = 1.0 - mismatched / max(eligible, 1)
    if fidelity < 0.95:
        raise SystemExit(
            f"replay/policy fidelity {fidelity:.1%} is below 95%; refusing to "
            "call these frozen-v5 trajectories")
    queue = curate(candidates, args.size, args.per_replay, args.per_bucket)
    if len(queue) < min(args.size, 100):
        raise SystemExit(
            f"only {len(queue)} diverse candidates survived; need at least 100")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in queue:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    manifest = {
        "experiment": "E3",
        "queue": str(out.relative_to(ROOT)).replace("\\", "/"),
        "source": args.replays,
        "player": args.player,
        "net": args.net,
        "net_sha256": sha256(net_path),
        "replay_files": len(paths),
        "eligible_decisions": eligible,
        "policy_mismatches": mismatched,
        "policy_fidelity": fidelity,
        "rankable_candidates": len(candidates),
        "queue_items": len(queue),
        "per_replay_cap": args.per_replay,
        "per_bucket_cap": args.per_bucket,
        "selection": "ascending selected/unselected boundary margin; "
                     "normalized entropy tie-break; exact option ties excluded",
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    buckets = Counter((x["select_type"], x["context"]) for x in queue)
    print(f"E3_QUEUE_OK {len(queue)} items from {len(paths)} replays")
    print(f"  fidelity={fidelity:.3%} ({mismatched}/{eligible} mismatches)")
    print(f"  candidates={len(candidates)} errors={errors}")
    print(f"  margin=[{min(x['boundary_margin'] for x in queue):.4f}, "
          f"{max(x['boundary_margin'] for x in queue):.4f}]")
    print(f"  buckets={dict(sorted(buckets.items()))}")
    print(f"  queue={out}")
    print(f"  manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
