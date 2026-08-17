"""Collect bc:all@grimmsnarl vs Ogerpon data and build an Ogerpon-resistant policy.

    python -X utf8 scripts/p101_ogerpon_resist.py record --games 150
    python -X utf8 scripts/p101_ogerpon_resist.py analyze
    python -X utf8 scripts/p101_ogerpon_resist.py arena --matches 1000
    python -X utf8 scripts/p101_ogerpon_resist.py screen --matches 500
    python -X utf8 scripts/p101_ogerpon_resist.py generate --games 1500
    python -X utf8 scripts/p101_ogerpon_resist.py finetune
    python -X utf8 scripts/p101_ogerpon_resist.py test --matches 500
    python -X utf8 scripts/p101_ogerpon_resist.py iterate --games 10000 --workers 4

Uses bc:all (policy_v5_s2_all, rules off) on both sides unless overridden.
Deck-tech variants were screened and underperformed stock grimmsnarl; the
resistant build is a policy fine-tune on grimmsnarl-vs-Ogerpon self-play shards.
`iterate` generates from the current best net, clones winning seats only, and
re-arenas vs Ogerpon. Target: 0.25 WR.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

NET_ALL = "out/policy_v5_s2_all.npz"
NET_OGER_FT = "out/policy_v5_s2_oger_ft.npz"
NET_BEST = "out/policy_v5_s2_oger_ft2.npz"
PDS_OGER = "artifacts/pds_ogerpon_matchup"
PDS_R2 = "artifacts/pds_ogerpon_r2"
PDS_ANCHOR = "artifacts/pds_all"
POLICY_ARCH = ["--opt-cols", "37", "--state-h", "512,256", "--head-h", "256,128",
               "--pool", "--loss", "listwise"]
RULES_OFF = "noChip,noSpread,noSrc"


def _a_spec(net: str = NET_ALL) -> str:
    return f"bc:grim,net={net},{RULES_OFF}"


A_SPEC = _a_spec()
B_SPEC = f"bc:oger,net={NET_ALL},{RULES_OFF}"
DECK_A = "grimmsnarl"
DECK_B = "teal_mask_ogerpon"
REPLAY_DIR = ROOT / "out" / "replays" / "p101_grimmsnarl_vs_ogerpon"
LOG_DIR = ROOT / "out" / "logs"
SUMMARY = LOG_DIR / "p101_ogerpon_resist_summary.txt"

TEAL_MASK = 96
SCORE_RE = re.compile(
    r"score=([\d.]+) \[([\d.]+), ([\d.]+)\].*over (\d+) games")


def _run(cmd: list[str], log_path: Path | None = None) -> subprocess.CompletedProcess:
    print(" ".join(cmd), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            return subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                                  stderr=subprocess.STDOUT, check=False)
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=False)


def cmd_record(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "p20_record_games.py"),
        "--a", A_SPEC, "--b", B_SPEC,
        "--deck-a", DECK_A, "--deck-b", DECK_B,
        "--games", str(args.games), "--swap",
        "--out", str(REPLAY_DIR),
    ]
    if args.no_obs:
        cmd.append("--no-obs")
    rc = _run(cmd).returncode
    if rc == 0:
        print(f"replays -> {REPLAY_DIR}", flush=True)
    return rc


def _load_replays() -> list[tuple[Path, dict]]:
    from p72_loss_autopsy import _records  # noqa: E402
    games = []
    if not REPLAY_DIR.exists():
        return games
    for path in sorted(REPLAY_DIR.glob("game*.json")):
        try:
            rep = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        games.append((path, rep))
    return games


def cmd_analyze(_args: argparse.Namespace) -> int:
    from p72_loss_autopsy import _nm, _pk  # noqa: E402
    from ptcg.env import sdk  # noqa: E402
    from sa import cards as cdb  # noqa: E402

    sdk.load()

    games = _load_replays()
    if not games:
        raise SystemExit(f"no replays in {REPLAY_DIR}; run record first")

    wins, losses, draws = 0, 0, 0
    missed_ko = 0
    opp_trainer_names = Counter()
    loss_turns: list[int] = []
    win_turns: list[int] = []

    for path, rep in games:
        rewards = rep.get("rewards") or [None, None]
        names = (rep.get("info") or {}).get("TeamNames") or []
        # seat 0 is bc:grim on even games (record --swap); use agent label
        grim_seat = 0
        for i, n in enumerate(names):
            if str(n).startswith("bc:grim"):
                grim_seat = i
                break
        if rewards[0] is None or rewards[1] is None:
            continue
        turns = len(rep.get("steps") or [])
        if rewards[grim_seat] > rewards[1 - grim_seat]:
            wins += 1
            win_turns.append(turns)
            a_won = True
        elif rewards[grim_seat] < rewards[1 - grim_seat]:
            losses += 1
            loss_turns.append(turns)
            a_won = False
        else:
            draws += 1
            continue

        vis = (rep.get("steps") or [[{}]])[0][0].get("visualize") or []
        for v in vis:
            if not isinstance(v, dict):
                continue
            ob = v.get("obs") or v
            state = ob.get("current") or {}
            sel = state.get("select") or ob.get("select")
            if not sel:
                continue
            me = state.get("yourIndex", 0)
            opp = 1 - int(me)
            for opt in sel.get("option") or []:
                if opt.get("playerIndex") != opp:
                    continue
                cid = int(opt.get("id") or 0)
                if cid <= 0:
                    continue
                try:
                    cat = str(cdb.card(cid).get("category") or "")
                except Exception:  # noqa: BLE001
                    cat = ""
                if cat in ("Item", "Supporter", "Stadium"):
                    opp_trainer_names[_nm(cid)] += 1

            ctx = sel.get("context")
            if ctx not in (13, 14, 15) or int(me) != grim_seat or a_won:
                continue
            for opt in sel.get("option") or []:
                if opt.get("playerIndex") != opp:
                    continue
                pk = _pk(state, opt)
                if not pk or int(pk.get("id") or 0) != TEAL_MASK:
                    continue
                hp = int(pk.get("hp") or 0)
                dmg = int(opt.get("damage") or 0)
                if 0 < hp <= 40 and dmg >= hp:
                    missed_ko += 1
                    break

    lines = [
        f"=== p101 autopsy {REPLAY_DIR.name} ===",
        f"games={len(games)}  grimmsnarl_w={wins} L={losses} D={draws}  "
        f"WR={wins / max(wins + losses, 1):.3f}",
        f"avg turns  win={sum(win_turns)/max(len(win_turns),1):.1f}  "
        f"loss={sum(loss_turns)/max(len(loss_turns),1):.1f}",
        f"missed low-HP Teal Mask KO lines (losses only scan): {missed_ko}",
        "",
        "opponent Item plays seen (all games):",
    ]
    for name, n in opp_trainer_names.most_common(12):
        lines.append(f"  {n:4d}  {name}")

    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    SUMMARY.write_text(text, encoding="utf-8")
    (LOG_DIR / "p101_ogerpon_autopsy.txt").write_text(text, encoding="utf-8")
    return 0


def _arena(deck_a: str, deck_b: str, b_spec: str, matches: int,
           archive: Path, log: Path) -> tuple[float, float, float, int]:
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "arena.py"), "play",
        A_SPEC, b_spec,
        "--deck-a", deck_a, "--deck-b", deck_b,
        "--matches", str(matches),
        "--archive", str(archive),
    ]
    proc = _run(cmd, log)
    if proc.returncode != 0:
        raise SystemExit(f"arena failed exit={proc.returncode} log={log}")
    body = log.read_text(encoding="utf-8", errors="replace")
    for line in body.splitlines():
        m = SCORE_RE.search(line)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))
    raise SystemExit(f"no score in {log}")


def _gen_cmd(net: str, games: int, seed: int, gid_base: int, out: str,
             tau: float) -> list[str]:
    return [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "p26_selfplay_gen.py"),
        "--net", net,
        "--deck", DECK_A,
        "--deck-b", DECK_B,
        "--opp", B_SPEC,
        "--tau", str(tau),
        "--games", str(games),
        "--seed", str(seed),
        "--gid-base", str(gid_base),
        "--out", out,
    ]


def cmd_generate(args: argparse.Namespace) -> int:
    out = getattr(args, "out", None) or PDS_OGER
    workers = max(1, int(getattr(args, "workers", 1)))
    if workers == 1:
        return _run(_gen_cmd(args.net, args.games, args.seed, args.gid_base,
                             out, args.tau)).returncode
    per = args.games // workers
    leftover = args.games - per * workers
    procs: list[subprocess.Popen] = []
    for i in range(workers):
        n = per + (leftover if i == workers - 1 else 0)
        shard_out = f"{out}/w{i}"
        cmd = _gen_cmd(args.net, n, args.seed + 1000 * i,
                       args.gid_base + i * 10_000, shard_out, args.tau)
        log = LOG_DIR / f"p101_generate_w{i}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(" ".join(cmd), flush=True)
        lf = log.open("w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf,
                                      stderr=subprocess.STDOUT))
    rc = 0
    for p in procs:
        rc |= p.wait()
    print(f"generate workers={workers} games={args.games} rc={rc} -> {out}",
          flush=True)
    return rc


def cmd_finetune(args: argparse.Namespace) -> int:
    ds = getattr(args, "ds", None) or PDS_OGER
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "train_policy.py"),
        "--ds", ds,
        *POLICY_ARCH,
        "--init", args.net,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--export-last",
        "--out", args.out,
        "--device", getattr(args, "device", "cpu"),
    ]
    if getattr(args, "winners_only", False):
        cmd.append("--winners-only")
    else:
        cmd.extend(["--advantage", str(args.advantage),
                    "--margin-max", str(args.margin_max)])
    if args.anchor_ds:
        cmd.extend(["--anchor-ds", args.anchor_ds, "--anchor-w", str(args.anchor_w)])
    rc = _run(cmd, LOG_DIR / "p101_ogerpon_finetune.log").returncode
    if rc == 0:
        print(f"fine-tuned policy -> {args.out}", flush=True)
    return rc


def cmd_test(args: argparse.Namespace) -> int:
    net = args.net
    a_spec = _a_spec(net)
    tag = Path(net).stem.replace("policy_", "")
    log = LOG_DIR / f"p101_{tag}_vs_ogerpon.txt"
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "arena.py"), "play",
        a_spec, B_SPEC,
        "--deck-a", DECK_A, "--deck-b", DECK_B,
        "--matches", str(args.matches),
        "--archive", str(ROOT / "out" / "arena" / f"p101_{tag}_vs_ogerpon.jsonl"),
    ]
    proc = _run(cmd, log)
    if proc.returncode != 0:
        return proc.returncode
    body = log.read_text(encoding="utf-8", errors="replace")
    for line in body.splitlines():
        m = SCORE_RE.search(line)
        if m:
            sc, lo, hi, n = float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))
            summary = (
                f"=== p101 test {net} @ grimmsnarl vs Ogerpon ===\n"
                f"grimmsnarl WR={sc:.3f} [{lo:.3f},{hi:.3f}] n={n}\n"
            )
            print(summary, flush=True)
            with SUMMARY.open("a", encoding="utf-8") as sf:
                sf.write(summary)
            return 0
    raise SystemExit(f"no score in {log}")


def cmd_arena(args: argparse.Namespace) -> int:
    log = LOG_DIR / "p101_grimmsnarl_vs_ogerpon_arena.txt"
    sc, lo, hi, n = _arena(
        DECK_A, DECK_B, B_SPEC, args.matches,
        ROOT / "out" / "arena" / "p101_grimmsnarl_vs_ogerpon.jsonl", log)
    line = (f"bc:all@grimmsnarl vs bc:oger@teal_mask_ogerpon  "
            f"{sc:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")
    print(line, flush=True)
    with SUMMARY.open("a", encoding="utf-8") as sf:
        sf.write(line + "\n")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    variants = [
        ("grimmsnarl", "stock 60"),
        ("grimmsnarl_anti_ogerpon", "Budew + Xerosic + Boss/Scrapper"),
        ("grimmsnarl_budew", "existing Budew variant"),
        ("grimmsnarl_xerosic", "existing Xerosic variant"),
    ]
    lines = [f"=== p101 screen vs Ogerpon + mirror, matches={args.matches} ==="]
    rows = []
    for deck, note in variants:
        tag = deck.replace("grimmsnarl", "grim").replace("_", "")
        log_o = LOG_DIR / f"p101_{tag}_vs_oger.txt"
        sc_o, lo_o, hi_o, n_o = _arena(
            deck, DECK_B, B_SPEC, args.matches,
            ROOT / "out" / "arena" / f"p101_{tag}_vs_oger.jsonl", log_o)
        log_m = LOG_DIR / f"p101_{tag}_mirror.txt"
        sc_m, lo_m, hi_m, n_m = _arena(
            deck, DECK_A, A_SPEC, args.matches // 2,
            ROOT / "out" / "arena" / f"p101_{tag}_mirror.jsonl", log_m)
        rows.append((deck, note, sc_o, lo_o, hi_o, sc_m, lo_m, hi_m))
        lines.append(
            f"{deck:28s}  vs oger {sc_o:.3f} [{lo_o:.3f},{hi_o:.3f}]  "
            f"mirror {sc_m:.3f} [{lo_m:.3f},{hi_m:.3f}]  ({note})")

    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    (LOG_DIR / "p101_ogerpon_screen.txt").write_text(text, encoding="utf-8")
    best = max(rows, key=lambda r: r[2])
    lines.append(f"\nBest vs Ogerpon: {best[0]} @ {best[2]:.3f}")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def cmd_iterate(args: argparse.Namespace) -> int:
    """One policy-iteration round: generate from --net, clone wins, arena."""
    gen = argparse.Namespace(
        net=args.net, games=args.games, seed=args.seed, gid_base=args.gid_base,
        tau=args.tau, workers=args.workers, out=args.ds)
    print(f"=== iterate generate {args.games} from {args.net} -> {args.ds} ===",
          flush=True)
    rc = cmd_generate(gen)
    if rc:
        return rc
    prior = PDS_OGER if (ROOT / PDS_OGER).exists() else ""
    ds = ",".join(p for p in (prior, args.ds) if p)
    ft = argparse.Namespace(
        net=args.net, out=args.out, epochs=args.epochs, lr=args.lr,
        ds=ds, winners_only=True, advantage=0.5, margin_max=2.0,
        anchor_ds=args.anchor_ds, anchor_w=args.anchor_w, device=args.device)
    print(f"=== iterate finetune winners-only ds={ds} -> {args.out} ===",
          flush=True)
    rc = cmd_finetune(ft)
    if rc:
        return rc
    print(f"=== iterate test {args.out} vs Ogerpon matches={args.matches} ===",
          flush=True)
    return cmd_test(argparse.Namespace(net=args.out, matches=args.matches))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="record watchable replays")
    p_rec.add_argument("--games", type=int, default=150)
    p_rec.add_argument("--no-obs", action="store_true")
    p_rec.set_defaults(func=cmd_record)

    p_an = sub.add_parser("analyze", help="autopsy recorded games")
    p_an.set_defaults(func=cmd_analyze)

    p_ar = sub.add_parser("arena", help="large paired arena (no full replays)")
    p_ar.add_argument("--matches", type=int, default=1000)
    p_ar.set_defaults(func=cmd_arena)

    p_sc = sub.add_parser("screen", help="A/B deck variants vs Ogerpon + mirror")
    p_sc.add_argument("--matches", type=int, default=500)
    p_sc.set_defaults(func=cmd_screen)

    p_gen = sub.add_parser("generate", help="self-play shards grimmsnarl vs Ogerpon")
    p_gen.add_argument("--games", type=int, default=1500)
    p_gen.add_argument("--net", default=NET_ALL)
    p_gen.add_argument("--tau", type=float, default=0.4)
    p_gen.add_argument("--seed", type=int, default=101)
    p_gen.add_argument("--gid-base", type=int, default=910000)
    p_gen.add_argument("--workers", type=int, default=1)
    p_gen.add_argument("--out", default=PDS_OGER)
    p_gen.set_defaults(func=cmd_generate)

    p_ft = sub.add_parser("finetune", help="fine-tune on matchup shards + BC anchor")
    p_ft.add_argument("--net", default=NET_ALL)
    p_ft.add_argument("--out", default=NET_OGER_FT)
    p_ft.add_argument("--ds", default=PDS_OGER)
    p_ft.add_argument("--epochs", type=int, default=3)
    p_ft.add_argument("--lr", type=float, default=5e-4)
    p_ft.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p_ft.add_argument("--winners-only", action="store_true")
    p_ft.add_argument("--anchor-ds", default="",
                      help="optional anchor corpus (pds_all needs ~1GB RAM)")
    p_ft.add_argument("--anchor-w", type=float, default=1.0)
    p_ft.add_argument("--advantage", type=float, default=0.5)
    p_ft.add_argument("--margin-max", type=float, default=2.0)
    p_ft.set_defaults(func=cmd_finetune)

    p_test = sub.add_parser("test", help="arena test fine-tuned grimmsnarl vs Ogerpon")
    p_test.add_argument("--net", default=NET_BEST)
    p_test.add_argument("--matches", type=int, default=500)
    p_test.set_defaults(func=cmd_test)

    p_it = sub.add_parser("iterate", help="generate from net, clone wins, arena")
    p_it.add_argument("--net", default=NET_BEST)
    p_it.add_argument("--out", default="out/policy_v5_s2_oger_r2.npz")
    p_it.add_argument("--ds", default=PDS_R2)
    p_it.add_argument("--games", type=int, default=10000)
    p_it.add_argument("--workers", type=int, default=4)
    p_it.add_argument("--tau", type=float, default=0.4)
    p_it.add_argument("--seed", type=int, default=202)
    p_it.add_argument("--gid-base", type=int, default=930000)
    p_it.add_argument("--epochs", type=int, default=5)
    p_it.add_argument("--lr", type=float, default=1e-3)
    p_it.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p_it.add_argument("--anchor-ds", default="")
    p_it.add_argument("--anchor-w", type=float, default=1.0)
    p_it.add_argument("--matches", type=int, default=500)
    p_it.set_defaults(func=cmd_iterate)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
