# Private E1 GPU runbook

The prepared archive is:

```text
out/e1/e1_gpu_bundle.zip
```

It is about 15.4 MB and contains the licensed `cg` engine. Keep the Kaggle
dataset and notebook private; do not publish the archive or notebook output
containing it.

## Kaggle setup

1. Create a private Kaggle dataset containing only `e1_gpu_bundle.zip`.
2. Create a private GPU notebook and attach that dataset.
3. Enable a GPU accelerator.
4. Run the following cell:

```python
from pathlib import Path
import glob
import shutil
import subprocess
import zipfile

matches = glob.glob("/kaggle/input/**/e1_gpu_bundle.zip", recursive=True)
assert len(matches) == 1, matches

work = Path("/kaggle/working/e1")
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
with zipfile.ZipFile(matches[0]) as zf:
    zf.extractall(work)

subprocess.run(
    ["python", "-X", "utf8", "scripts/p39_multitask_smoke.py"],
    cwd=work,
    check=True,
)
subprocess.run(
    [
        "python", "-X", "utf8", "scripts/p40_e1_sweep.py",
        "--device", "cuda",
    ],
    cwd=work,
    check=True,
)
shutil.make_archive(
    "/kaggle/working/e1_results",
    "zip",
    root_dir=work / "out" / "e1",
)
print("Download /kaggle/working/e1_results.zip")
```

Expected outputs:

```text
out/e1/
  control_seed0.npz
  outcome_seed0.npz
  count_seed0.npz
  both_seed0.npz
  control_seed0.log
  outcome_seed0.log
  count_seed0.log
  both_seed0.log
  manifest.json
```

Download `e1_results.zip` and place it in the repository at
`out/e1/e1_results.zip`. Do not rename or unpack it; the local intake step will
verify the manifest and hashes before arena testing.
