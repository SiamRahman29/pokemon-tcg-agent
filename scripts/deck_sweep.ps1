# Which deck can our policy clone actually pilot? Same agent, same opponent
# (rule:iono), one run per deck. crispin_box already measured at 0.068 and
# grimmsnarl at 0.480 -- a deck's meta win-rate does not predict this.
# NB: do NOT name a param $Matches -- it collides with PowerShell's automatic
# regex variable and every assignment throws ArgumentTransformationMetadata.
param([string]$Root = 'E:\Kaggle\pokemon-tcg-simulation-2', [int]$Pairs = 150)

foreach ($d in @('iono','dragapult_ex','mega_abomasnow_ex','mega_lucario_ex','alakazam')) {
  Write-Output "##### deck=$d"
  & python -X utf8 -u "$Root\scripts\arena.py" play bc rule:iono `
      --deck-a $d --deck-b iono --matches $Pairs `
      --archive "out/arena/deck_$d.jsonl" 2>&1 |
    Where-Object { $_ -notmatch '^\s*match ' -and $_ -notmatch 'No Basic Pokemon' }
}
