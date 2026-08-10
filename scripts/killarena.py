"""Kill stray arena/collection processes.

⚠ Lives in a FILE rather than `python -c` on purpose. A `-c` killer carries its
own match pattern in its command line, and so does the shell that launched it —
so a pattern like "arena.py" matches the enclosing bash and SIGTERMs the very
command that is starting the next run. That happened twice here; the second
time it killed an A/B launch at t=0.
"""
import sys

import psutil

PAT = sys.argv[1] if len(sys.argv) > 1 else "arena.py"
me = psutil.Process()
mine = {me.pid} | {p.pid for p in me.parents()}
n = 0
for p in psutil.process_iter(["pid", "cmdline", "name"]):
    if p.info["pid"] in mine:
        continue
    try:
        cl = " ".join(p.info["cmdline"] or "")
    except Exception:
        continue
    if PAT in cl and "killarena" not in cl:
        try:
            p.kill()
            print("killed", p.info["pid"])
            n += 1
        except Exception as e:
            print("fail", p.info["pid"], e)
print("killed", n)
