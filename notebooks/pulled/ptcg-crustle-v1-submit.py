from pathlib import Path
import base64
import gzip
import glob
import importlib.util
import shutil
import sys
import tarfile

MAIN_GZ_B64 = """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"""
DECK_TEXT = """344
344
344
344
345
345
345
345
1147
1147
1147
1147
1159
1264
1264
1264
1264
1212
1212
1212
1212
1224
1224
1224
1224
18
18
18
18
11
11
11
11
1086
1086
1086
1086
14
14
14
14
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
"""

Path("main.py").write_text(gzip.decompress(base64.b64decode(MAIN_GZ_B64)).decode("utf-8"), encoding="utf-8")
Path("deck.csv").write_text(DECK_TEXT, encoding="utf-8")

cg_path = None
for pattern in [
    "/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/cg",
    "/kaggle/input/**/sample_submission/cg",
    "/kaggle/input/**/cg",
    str(Path.cwd().parent.parent / "sample_submission" / "cg"),
]:
    hits = glob.glob(pattern, recursive=True)
    if hits:
        cg_path = Path(hits[0])
        break
if cg_path is None:
    raise FileNotFoundError("Could not locate competition cg folder")

if Path("cg").exists():
    shutil.rmtree("cg")
shutil.copytree(cg_path, "cg")

def add_clean_dir(tar, source, arcname):
    source = Path(source)
    for path in source.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        tar.add(path, arcname=str(Path(arcname) / path.relative_to(source)))

with tarfile.open("submission.tar.gz", "w:gz") as tar:
    tar.add("main.py", arcname="main.py")
    tar.add("deck.csv", arcname="deck.csv")
    add_clean_dir(tar, "cg", "cg")

with tarfile.open("submission.tar.gz", "r:gz") as tar:
    names = tar.getnames()
print("created submission.tar.gz")
print("files", len(names))
print("top", [n for n in names if n in ("main.py", "deck.csv") or n.startswith("cg/")][:10])

sys.path.insert(0, str(Path.cwd()))
spec = importlib.util.spec_from_file_location("agent_module", "main.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("deck length", len(mod.agent({"select": None, "logs": [], "current": None})))
