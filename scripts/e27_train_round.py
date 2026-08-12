"""E27: the three post-generation steps of one round, in one command.

`scripts/kaggle/launch.py --then` runs exactly ONE post-command after the
generation shards finish, and a round needs three steps in order. This is that
command, so the round recipe lives in one auditable place rather than being
re-typed per round:

    train_value  -> V_r on THIS round's own games (policy evaluation)
    p92          -> per-decision TD advantage, reconciled before it is written
    train_policy -> pi_{r+1} = pi_r fine-tuned on that advantage

🔴 **Why this runs on Kaggle rather than locally.** `train_policy.Data`
materialises the whole corpus and concatenates, so a 1.5M-row round peaks near
3.7 GB against 4.0 GB free on this box -- B8's documented binding constraint
(§8ao: "the binding constraint was memory, not compute") arriving again, one
round later. Round 1 fitted; round 2 was OOM-killed during load, three times,
with an empty log because it died before the first print. A Kaggle kernel has
~30 GB.

⚠ Every hyperparameter here is the one frozen in
`docs/experiments/E27-policy-iteration.md`. **If you find yourself editing a
number in this file, you are changing a pre-registered experiment** -- the doc
forbids sweeping beta, the clip, or the architecture.

    python -X utf8 scripts/e27_train_round.py --data artifacts/e27_r2 \\
        --init out/policy_e27_r1.npz --tag e27r2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Frozen in the pre-registration. Named here so a diff shows a change to them.
ARCH = ["--opt-cols", "37", "--state-h", "512,256", "--head-h", "256,128",
        "--pool", "--loss", "listwise", "--bs", "1024"]
BETA = "0.5"
EPOCHS = "3"
LR = "2e-4"
VALUE_EPOCHS = "30"


def run(step: str, cmd: list[str]) -> None:
    print(f"\n=== {step} ===\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run([sys.executable, "-u", "-X", "utf8"] + cmd, cwd=ROOT)
    if r.returncode != 0:
        # Fail loudly and STOP. A round whose value net failed but whose policy
        # trained anyway would produce a net that looks ordinary and carries a
        # meaningless advantage -- the shape of every "plausible number, not a
        # crash" defect in this repo.
        raise SystemExit(f"{step} failed with exit {r.returncode}; "
                         f"the round is ABANDONED rather than continued")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the round's corpus dir")
    ap.add_argument("--init", required=True, help="pi_r, the policy to improve")
    ap.add_argument("--tag", required=True, help="e.g. e27r2")
    args = ap.parse_args()

    # Written under out/e27/ so a Kaggle job can `--collect out/e27` and pull
    # back only what the round PRODUCED, not the ~117 MB of input nets the
    # payload also unpacks into out/.
    (ROOT / "out" / "e27").mkdir(parents=True, exist_ok=True)
    vnet = f"out/e27/value_{args.tag}.npz"
    pnet = f"out/e27/policy_{args.tag}.npz"

    run("1/3 policy evaluation (V on this round's own games)",
        ["scripts/train_value.py", "--data", args.data, "--out", vnet,
         "--epochs", VALUE_EPOCHS, "--split-seed", "0"])

    run("2/3 per-decision TD advantage (reconciled before writing)",
        ["scripts/p92_td_advantage.py", "--data", args.data, "--value", vnet])

    run("3/3 policy improvement",
        ["scripts/train_policy.py", "--ds", args.data, "--init", args.init,
         *ARCH, "--advantage-col", BETA, "--epochs", EPOCHS, "--lr", LR,
         "--export-last", "--out", pnet])

    print(f"\n=== round {args.tag} complete: {vnet}, {pnet} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
