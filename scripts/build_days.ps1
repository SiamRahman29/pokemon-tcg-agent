# Build POLICY shards for new replay days. The value net is excluded from the
# shipped agent (arena says it does not help), so we no longer build ds/ too --
# that halves the work. Add -WithValue if that ever changes.
param(
  [string]$Root = 'E:\Kaggle\pokemon-tcg-simulation-2',
  [string[]]$Days = @('2026-07-27','2026-07-19','2026-07-18','2026-07-17'),
  [switch]$WithValue
)

foreach ($d in $Days) {
  $dir = "$Root\replays\$d"
  if (-not (Test-Path $dir)) { Write-Output "##### $d MISSING - skip"; continue }
  $n = (Get-ChildItem $dir -Filter *.json).Count
  if ($n -eq 0) { Write-Output "##### $d EMPTY - skip"; continue }
  $tag = 'd' + $d.Substring(8, 2)
  Write-Output "##### $d -> pds/$tag ($n replays)"
  & python -X utf8 -u "$Root\scripts\build_policy_dataset.py" --out "artifacts/pds/$tag" $dir 2>&1 |
    Select-String -NotMatch 'No Basic Pokemon' | Select-Object -Last 3
  if ($WithValue) {
    Write-Output "##### $d -> ds/$tag"
    & python -X utf8 -u "$Root\scripts\build_dataset.py" --out "artifacts/ds/$tag" --stride 1 $dir 2>&1 |
      Select-String -NotMatch 'No Basic Pokemon' | Select-Object -Last 3
  }
}
Write-Output "BUILD_DAYS_DONE"
