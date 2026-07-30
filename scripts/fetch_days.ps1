# Fetch several replay days in sequence. Network-bound, so it costs almost no
# CPU and can run alongside arenas/training. Idempotent per fetch_top_episodes.
param(
  [string]$Root = 'E:\Kaggle\pokemon-tcg-simulation-2',
  [int]$Max = 400,
  # NB: `powershell -File this.ps1 -Dates a,b` does not bind a real array --
  # edit this default and launch with no arguments.
  # Last day on disk is 2026-07-27; HANDOFF §3.1 needs these three.
  [string[]]$Dates = @('2026-07-30','2026-07-29','2026-07-28')
)

foreach ($d in $Dates) {
  Write-Output "##### $d"
  & python -X utf8 -u "$Root\scripts\fetch_top_episodes.py" --date $d --max $Max
}
