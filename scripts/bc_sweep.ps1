# BC-only matchups are ~0.17s/game, so we can measure them at n>=400 instead of
# n=24. Runs sequentially in ONE process to keep the search arenas' CPU free.
param([string]$Root = 'E:\Kaggle\pokemon-tcg-simulation-2')

$runs = @(
  @{ tag='bc_iono_1k';      a='bc'; b='rule:iono';        da='grimmsnarl';  db='iono';               m=500 },
  @{ tag='bc_dragapult';    a='bc'; b='rule:dragapult';   da='grimmsnarl';  db='dragapult_ex';       m=200 },
  @{ tag='bc_abomasnow';    a='bc'; b='rule:abomasnow';   da='grimmsnarl';  db='mega_abomasnow_ex';  m=200 },
  @{ tag='bc_lucario';      a='bc'; b='rule:lucario';     da='grimmsnarl';  db='mega_lucario_ex';    m=200 },
  @{ tag='bc_crispin_iono'; a='bc'; b='rule:iono';        da='crispin_box'; db='iono';               m=200 },
  @{ tag='bc_alakazam_iono';a='bc'; b='rule:iono';        da='alakazam';    db='iono';               m=200 },
  @{ tag='bc_random';       a='bc'; b='random';           da='grimmsnarl';  db='iono';               m=100 }
)

foreach ($r in $runs) {
  Write-Output "##### $($r.tag)"
  & python -X utf8 -u "$Root\scripts\arena.py" play $r.a $r.b `
      --deck-a $r.da --deck-b $r.db --matches $r.m `
      --archive "out/arena/$($r.tag).jsonl" 2>&1 |
    Where-Object { $_ -notmatch '^\s*match ' -and $_ -notmatch 'No Basic Pokemon' }
}
