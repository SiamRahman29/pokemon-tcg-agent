# Which deck can our policy clone actually pilot?  Same agent, same opponent,
# one run per deck.
#
# The opponent now defaults to rule:v10,noS (the public LB-950 agent), NOT the
# old rule:iono. Anchoring on rule:iono measures almost nothing: rule:lucario
# and the ~104-Elo-stronger rule:v10 both score ~0.78 against it, because that
# number is the lucario deck beating the iono deck -- the pilot is invisible.
# See HANDOFF rule 8. Every ranking this script produced before 2026-07-28 was
# measured against rule:iono and is superseded.
#
# NB: do NOT name a param $Matches -- it collides with PowerShell's automatic
# regex variable and every assignment throws ArgumentTransformationMetadata.
param(
  [string]$Root = 'E:\Kaggle\pokemon-tcg-simulation-2',
  [int]$Pairs = 200,
  [string]$Opponent = 'rule:v10,noS',
  [string]$OppDeck = 'lucario_v10',
  [string]$Tag = 'v10'
)

foreach ($d in @('grimmsnarl','iono','dragapult_ex','mega_abomasnow_ex','mega_lucario_ex','alakazam','crispin_box')) {
  Write-Output "##### deck=$d vs $Opponent"
  & python -X utf8 -u "$Root\scripts\arena.py" play bc $Opponent `
      --deck-a $d --deck-b $OppDeck --matches $Pairs `
      --archive "out/arena/deck_${Tag}_$d.jsonl" 2>&1 |
    Where-Object { $_ -notmatch '^\s*match ' -and $_ -notmatch 'No Basic Pokemon' }
}
