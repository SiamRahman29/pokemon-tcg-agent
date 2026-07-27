cd "E:/Kaggle/pokemon-tcg-simulation-2"
for d in 24 22 21; do
  echo "STEP ds/d$d"
  python -X utf8 scripts/build_dataset.py --out artifacts/ds/d$d --stride 1 replays/2026-07-$d 2>&1 | grep -vF "No Basic Pokemon." | tail -3
  echo "STEP pds/d$d"
  python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds/d$d replays/2026-07-$d 2>&1 | grep -vF "No Basic Pokemon." | tail -3
done
echo "NEWDAYS_BUILD_DONE"
