"""Build agents/sa/deck_library.json from mined replays + sample decks.

    python scripts/build_deck_library.py <replay_dir> [...]

Library = the most common exact 60-card lists (weight = times seen, winners
weighted extra) plus the four sample decks as a floor.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "agents"):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "agents" / "sa" / "deck_library.json"
MAX_DECKS = 40


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]]
    usage: Counter = Counter()
    for rep_dir in dirs:
        for path in sorted(rep_dir.glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                vis = d["steps"][0][0].get("visualize") or []
                decks = (vis[0]["action"] if vis
                         else [d["steps"][1][i]["action"] for i in range(2)])
                rewards = d["rewards"]
                for seat in (0, 1):
                    if not decks[seat] or len(decks[seat]) != 60:
                        continue
                    key = tuple(sorted(decks[seat]))
                    w = 2 if (rewards[seat] or 0) > (rewards[1 - seat] or 0) \
                        else 1
                    usage[key] += w
            except Exception:
                pass

    top = usage.most_common(MAX_DECKS)

    # sample decks as fallback entries with tiny weight
    import importlib
    for mod in ("iono", "dragapult_ex", "mega_abomasnow_ex",
                "mega_lucario_ex"):
        dl = importlib.import_module(f"decks.{mod}").DECKLIST
        key = tuple(sorted(c for c, n in dl.items() for _ in range(n)))
        if key not in dict(top):
            top.append((key, 1))

    entries = []
    for key, weight in top:
        counts = Counter(key)
        entries.append({"weight": weight,
                        "cards": {str(c): n for c, n in counts.items()}})
    OUT.write_text(json.dumps(entries), encoding="utf-8")
    print(f"wrote {len(entries)} decks -> {OUT} "
          f"(top weight {top[0][1] if top else 0})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
