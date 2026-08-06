"""Build policy-cloning shards from replay JSONs.

    python scripts/build_policy_dataset.py --out artifacts/pds/d26 replays/2026-07-26

One row per select with >=2 options: state features + per-option features +
multi-hot chosen mask (from the replay's actual action).

⚠ By default this clones BOTH seats of every game -- every archetype and every
skill level in the dump. `--player NAME` keeps only the seats belonging to the
named team(s), which is how an EXPERT corpus is built from a third-party dump:

    python scripts/build_policy_dataset.py --out artifacts/pds_expert \\
        --player "Raja Biswas" --player "Sixth Sense" \\
        replays/sixth_sense_31-07-2026

`--ratings` tags every row with the LB score of the demonstrator who made that
choice, so a corpus can be reweighted or sliced by demonstrator strength (B7):

    python scripts/build_policy_dataset.py --out artifacts/pds_v3r \\
        --ratings out/lb/pokemon-tcg-ai-battle.zip replays/2026-07-26
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa.features import attr_feats, extra_feats, featurize  # noqa: E402
from sa.optfeat import option_features, OPT_DENSE  # noqa: E402

SHARD_ROWS = 60_000
SEL_DENSE = 14
NO_RATING = np.float32("nan")


def load_ratings(path: Path) -> tuple[dict[str, float], dict[int, float]]:
    """Kaggle LB export -> (by team name, by teamId). Accepts the .zip that
    `competition_leaderboard_download` writes or the .csv inside it."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            if not names:
                raise SystemExit(f"{path}: no .csv inside")
            text = z.read(names[0]).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    by_name: dict[str, float] = {}
    by_id: dict[int, float] = {}
    by_user: dict[str, float] = {}
    n_teams = 0
    for row in csv.DictReader(text.splitlines()):
        try:
            score = float(row["Score"])
        except (TypeError, ValueError, KeyError):
            continue
        n_teams += 1
        by_name[row["TeamName"]] = score
        try:
            by_id[int(row["TeamId"])] = score
        except (TypeError, ValueError, KeyError):
            pass
        # A replay's TeamNames can hold a MEMBER's username rather than the
        # team's display name (teams merge and rename; `zoroark190` is how the
        # LB's #1 `James Cox & Henry Chao` appears in 07-26 replays). Exact
        # member matches are safe; ambiguous ones are dropped, not guessed.
        for user in (row.get("TeamMemberUserNames") or "").split(","):
            user = user.strip()
            if user:
                by_user[user] = score if user not in by_user else float("nan")
    # Display names always win; a username that collides with some other team's
    # display name, or with a second team, is dropped rather than resolved.
    n_user = 0
    for user, score in by_user.items():
        if user not in by_name and score == score:
            by_name[user] = score
            n_user += 1
    print(f"ratings: {n_teams} teams (+{n_user} member usernames) "
          f"from {path.name}")
    return by_name, by_id


def name_id(name: str) -> int:
    """Stable per-team id for dumps with no `episodes_meta.json` sidecar (the
    daily replay dirs). Deterministic across runs and processes, unlike
    `hash()`, so a corpus rebuilt tomorrow keeps the same ids."""
    import hashlib
    h = hashlib.sha1(name.encode("utf-8")).digest()
    return -int.from_bytes(h[:6], "big")   # negative = derived, not Kaggle's


def load_episode_meta(d: Path) -> dict[int, dict[int, tuple[int, int]]]:
    """`episodes_meta.json` -> {episode_id: {seat: (submissionId, teamId)}}.
    Only targeted per-team dumps carry it; day dumps return {}."""
    p = d / "episodes_meta.json"
    if not p.exists():
        return {}
    out: dict[int, dict[int, tuple[int, int]]] = {}
    for ep in json.loads(p.read_text(encoding="utf-8")):
        seats = {}
        for a in ep.get("agents") or []:
            seats[int(a.get("index") or 0)] = (int(a.get("submissionId") or -1),
                                               int(a.get("teamId") or -1))
        out[int(ep["id"])] = seats
    return out


def sel_features(sel: dict) -> np.ndarray:
    v = np.zeros(SEL_DENSE, dtype=np.float32)
    t = sel.get("type") or 0
    if t < 11:
        v[t] = 1.0
    v[11] = sel.get("minCount", 0) / 5.0
    v[12] = sel.get("maxCount", 0) / 5.0
    v[13] = (sel.get("context") or 0) / 50.0
    return v


class Writer:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        self.reset()

    def reset(self):
        self.sd, self.slots, self.seld, self.gid = [], [], [], []
        self.xd, self.xslots, self.attr = [], [], []
        self.bags = {"my_hand": [], "my_discard": [], "opp_discard": []}
        self.od, self.ocard, self.oatk, self.otgt, self.chosen = [], [], [], [], []
        self.off = [0]
        self.won = []
        self.rating, self.opp_rating, self.team_id, self.sub_id = [], [], [], []

    def add(self, dense, bags, seld, opts, chosen_mask, gid, won,
            rating=NO_RATING, opp_rating=NO_RATING, team_id=-1, sub_id=-1,
            extra=None, attr=None):
        self.sd.append(dense)
        # The v4 state block (features.extra_feats). Written unconditionally --
        # a trainer that does not want it simply does not read these arrays,
        # which is what makes the v3 control run on the IDENTICAL rows.
        xd, xids = extra
        self.xd.append(xd)
        self.xslots.append(xids)
        # The v6 card-attribute block (features.attr_feats), same contract:
        # always written, so a v5 control trains on the IDENTICAL rows.
        self.attr.append(attr)
        self.slots.append(bags["slots"])
        for k in self.bags:
            self.bags[k].append(bags[k])
        self.seld.append(seld)
        od, oc, oa, ot = opts
        self.od.append(od)
        self.ocard.append(oc)
        self.oatk.append(oa)
        self.otgt.append(ot)
        self.chosen.append(chosen_mask)
        self.off.append(self.off[-1] + len(oc))
        self.gid.append(gid)
        self.won.append(won)
        self.rating.append(rating)
        self.opp_rating.append(opp_rating)
        self.team_id.append(team_id)
        self.sub_id.append(sub_id)
        if len(self.sd) >= SHARD_ROWS:
            self.flush()

    def flush(self):
        if not self.sd:
            return
        arrs = {
            "dense": np.stack(self.sd),
            "slots": np.stack(self.slots),
            "seld": np.stack(self.seld),
            "xdense": np.stack(self.xd),
            "xslots": np.stack(self.xslots),
            "attr": np.stack(self.attr),
            "gid": np.asarray(self.gid, dtype=np.int64),
            "won": np.asarray(self.won, dtype=np.float32),
            # B7: who made this choice, and how good are they? NaN = the team
            # was not on the LB snapshot (renamed, or withdrawn).
            "rating": np.asarray(self.rating, dtype=np.float32),
            "opp_rating": np.asarray(self.opp_rating, dtype=np.float32),
            "team_id": np.asarray(self.team_id, dtype=np.int64),
            "sub_id": np.asarray(self.sub_id, dtype=np.int64),
            "opt_dense": np.concatenate(self.od),
            "opt_card": np.concatenate(self.ocard),
            "opt_attack": np.concatenate(self.oatk),
            "opt_target": np.concatenate(self.otgt),
            "opt_chosen": np.concatenate(self.chosen),
            "opt_off": np.asarray(self.off, dtype=np.int64),
        }
        for k, lists in self.bags.items():
            off = np.zeros(len(lists) + 1, dtype=np.int64)
            for i, a in enumerate(lists):
                off[i + 1] = off[i] + len(a)
            arrs[f"bag_{k}_flat"] = (np.concatenate(lists) if off[-1]
                                     else np.zeros(0, dtype=np.int32))
            arrs[f"bag_{k}_off"] = off
        path = self.out_dir / f"shard_{self.idx:03d}.npz"
        np.savez_compressed(path, **arrs)
        print(f"  wrote {path.name}: {len(self.sd)} rows")
        self.idx += 1
        self.reset()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--player", action="append", default=[],
                    help="keep only seats owned by this team name (repeatable). "
                         "Default: clone both seats of every game.")
    ap.add_argument("--players-file",
                    help="UTF-8 file of team names, one per line, added to "
                         "--player (for name lists the shell cannot quote)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="drop these team names even if --player/--players-file "
                         "names them. ⚠ Always exclude OURSELVES from a control "
                         "population: our own agent's selects are what the net "
                         "was fitted to, so leaving `Scio` in inflates "
                         "agreement toward 100%% for those rows.")
    ap.add_argument("--ratings",
                    help="Kaggle leaderboard export (.zip or .csv). Tags every "
                         "row with the demonstrator's LB score (B7). Without "
                         "it every row's rating is NaN.")
    ap.add_argument("--aliases", default="replays/team_aliases.tsv",
                    help="TSV of `replay name<TAB>LB team name` for teams that "
                         "renamed or merged and cannot be matched exactly. "
                         "Missing file = no aliases.")
    args = ap.parse_args()

    keep = set(args.player)
    if args.players_file:
        keep |= {ln.strip() for ln
                 in Path(args.players_file).read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#")}
    # ⚠ An empty filter means "clone both seats of every game" -- silently the
    # OPPOSITE of what was asked. An empty --players-file used to build a
    # whole-dump corpus under an expert corpus's name.
    if (args.player or args.players_file) and not keep:
        raise SystemExit("--player/--players-file given but resolved to zero "
                         "names; refusing to build an unfiltered corpus")
    drop = set(args.exclude)
    if drop:
        hit = keep & drop
        keep -= drop
        print(f"--exclude: dropped {sorted(hit)} from the demonstrator set")
        if (args.player or args.players_file) and not keep:
            raise SystemExit("--exclude removed every demonstrator")
    rate_name, rate_id = ({}, {})
    alias: dict[str, str] = {}
    team_name: dict[int, str] = {}
    ap_path = ROOT / args.aliases
    if ap_path.exists():
        for ln in ap_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip() or ln.startswith("#") or "\t" not in ln:
                continue
            old, new = (s.strip() for s in ln.split("\t", 1))
            alias[old] = new
    if args.ratings:
        rate_name, rate_id = load_ratings(Path(args.ratings))
        n_alias = 0
        for old, new in alias.items():
            if new in rate_name and old not in rate_name:
                rate_name[old] = rate_name[new]
                n_alias += 1
            elif new not in rate_name:
                print(f"  alias target not on the LB, ignored: {new!r}",
                      file=sys.stderr)
        print(f"  aliases: {n_alias} applied from {ap_path.name}")
    writer = Writer(ROOT / args.out)
    n_games = n_rows = n_err = n_skip_game = n_skip_seat = 0
    n_rated = n_unrated = 0
    unrated_names: dict[str, int] = {}
    for d in args.dirs:
        ep_meta = load_episode_meta(Path(d))
        for path in sorted(Path(d).glob("*.json")):
            if path.name == "manifest.json" or not path.stem.isdigit():
                continue   # sidecars: manifest.json, episodes_meta.json, ...
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
                rewards = rep["rewards"]
                if rewards[0] is None or rewards[1] is None:
                    continue
                names = (rep.get("info") or {}).get("TeamNames") or []
                seats = ({i for i, n in enumerate(names) if n in keep}
                         if keep else
                         ({i for i, n in enumerate(names) if n not in drop}
                          if drop else None))
                if seats is not None and not seats:
                    n_skip_game += 1
                    continue
                vis = rep["steps"][0][0].get("visualize") or []
                try:
                    gid = int(path.stem)
                except ValueError:
                    gid = hash(path.stem) & 0x7FFFFFFF
                n_games += 1
                # Per-seat identity. The sidecar's teamId is authoritative when
                # present -- `info.TeamNames` is a DISPLAY name and teams rename
                # mid-window (§8q: one demonstrator appeared as two).
                meta = ep_meta.get(gid, {})
                seat_rating: dict[int, float] = {}
                seat_team: dict[int, int] = {}
                seat_sub: dict[int, int] = {}
                for i in range(max(len(names), len(meta))):
                    sub, tid = meta.get(i, (-1, -1))
                    r = rate_id.get(tid) if tid >= 0 else None
                    nm = names[i] if i < len(names) else ""
                    if r is None and nm:
                        r = rate_name.get(nm)
                    seat_rating[i] = float(r) if r is not None else float("nan")
                    if tid < 0 and nm:
                        # No sidecar: identify the demonstrator by name. The
                        # alias file has already merged renames, so this does
                        # not split one team in two (§8q).
                        tid = name_id(alias.get(nm, nm))
                        team_name[tid] = alias.get(nm, nm)
                    elif tid >= 0 and nm:
                        team_name[tid] = nm
                    seat_team[i] = tid
                    seat_sub[i] = sub
                    if args.ratings:
                        if r is None:
                            n_unrated += 1
                            nm = names[i] if i < len(names) else f"seat{i}"
                            unrated_names[nm] = unrated_names.get(nm, 0) + 1
                        else:
                            n_rated += 1
                for v in vis:
                    obs = v.get("obs")
                    if not obs or not obs.get("current") or not obs.get("select"):
                        continue
                    state = obs["current"]
                    if state["result"] != -1:
                        continue
                    sel = obs["select"]
                    opts = sel.get("option") or []
                    if len(opts) < 2:
                        continue
                    action = v.get("selected")
                    if action is None:
                        action = v.get("action")
                    if not isinstance(action, list):
                        continue
                    picked = [a for a in action
                              if isinstance(a, int) and 0 <= a < len(opts)]
                    if len(picked) != len(action):
                        continue
                    me = state["yourIndex"]
                    if seats is not None and me not in seats:
                        n_skip_seat += 1
                        continue
                    won = 1.0 if rewards[me] > rewards[1 - me] else 0.0
                    dense, bags = featurize(state, me)
                    od = np.zeros((len(opts), OPT_DENSE), dtype=np.float32)
                    oc = np.zeros(len(opts), dtype=np.int32)
                    oa = np.zeros(len(opts), dtype=np.int32)
                    ot = np.zeros(len(opts), dtype=np.int32)
                    for i, o in enumerate(opts):
                        od[i], oc[i], oa[i], ot[i] = option_features(obs, o)
                    mask = np.zeros(len(opts), dtype=np.float32)
                    mask[picked] = 1.0
                    writer.add(dense, bags, sel_features(sel),
                               (od, oc, oa, ot), mask, gid, won,
                               extra=extra_feats(state, sel, me),
                               attr=attr_feats(state, me),
                               rating=seat_rating.get(me, float("nan")),
                               opp_rating=seat_rating.get(1 - me,
                                                          float("nan")),
                               team_id=seat_team.get(me, -1),
                               sub_id=seat_sub.get(me, -1))
                    n_rows += 1
            except Exception as exc:
                n_err += 1
                if n_err <= 5:
                    print(f"  {path.name}: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
    writer.flush()
    if team_name:
        # `team_id` in the shards is an int; this is how a report table gets a
        # name next to it. Merged with any existing map so a corpus built one
        # day-dir at a time accumulates rather than overwrites.
        tp = ROOT / args.out / "teams.json"
        old = (json.loads(tp.read_text(encoding="utf-8"))
               if tp.exists() else {})
        old.update({str(k): v for k, v in team_name.items()})
        tp.write_text(json.dumps(old, ensure_ascii=False, indent=1),
                      encoding="utf-8")
        print(f"  wrote {tp.name}: {len(old)} demonstrators")
    # ⚠ rule 9: a filter that matches nothing writes an empty corpus and exits 0.
    # A mistyped team name (CJK homoglyph, a rename) looks exactly like this.
    if keep and not n_games:
        raise SystemExit(f"player filter {sorted(keep)} matched ZERO games in "
                         f"{args.dirs}; check the exact team name")
    print(f"games={n_games} rows={n_rows} errors={n_err}")
    if args.ratings:
        tot = n_rated + n_unrated
        print(f"ratings: {n_rated}/{tot} seats matched the LB snapshot "
              f"({n_rated / max(tot, 1):.1%})")
        # ⚠ rule 9: an unmatched name is a SILENT NaN. Name the biggest misses
        # so a rename or an encoding mismatch cannot hide as "sparse data".
        for nm, c in sorted(unrated_names.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  unrated: {nm!r} x{c}")
    if keep:
        print(f"player filter {sorted(keep)}: skipped {n_skip_game} games with "
              f"no matching seat, {n_skip_seat} opponent-seat rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
