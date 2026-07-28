# Fetch several replay days in sequence. Network-bound, so it costs almost no
# CPU and can run alongside arenas/training. Idempotent per fetch_top_episodes.
param(
  [string]$Root = 'E:\Kaggle\pokemon-tcg-simulation-2',
  [int]$Max = 400,
  [string[]]$Dates = @('2026-07-27','2026-07-19','2026-07-18','2026-07-17')
)

foreach ($d in $Dates) {
  Write-Output "##### $d"
  & python -X utf8 -u "$Root\scripts\fetch_top_episodes.py" --date $d --max $Max
}
