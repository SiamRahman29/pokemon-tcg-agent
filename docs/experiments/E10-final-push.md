# E10 — THE FINAL PUSH (pre-registered 2026-08-07, day 25, before any cell ran)

**User directive (day 25, 2nd session): make a final push at the ~150-Elo gap to
the leaders before the freeze.** Sim closes **08-17**; last safe submission
**~08-15**; that leaves ≤7 working days. This file freezes the plan, the bars,
and the predictions BEFORE any experiment runs — the E9/E9b discipline.

**Incumbent:** `policy_v5_s2` (`55326513`, submitted 08-07 14:05, score pending).
**Reference opponent for screens:** `policy_v5_s1` (every §8bg screen used it).
**Shipped config everywhere:** `--no-rules` both arms (the §8be/§8bf trap).

---

## The strategic frame — three measured facts pick these experiments

1. **The climb runs through the mirror.** The opponent pool is a function of our
   own rating (§8ac): mirror = 33.3% of our games at 955, **51.1% above opponent
   rating 900, 71.4% above 1000**. Rank 129 → top-20 means beating agents on our
   exact 60.
2. **A field clone cannot win the mirror by being a better field clone.** Against
   field-modal play it converges to 0.500 by construction, and agreement with the
   field is *already* what predicts our strength (§8u). The leaders are
   handcrafted-rule agents (§0), i.e. they hold 1145–1166 by doing something in
   this matchup that the field mode does not. That difference is minable: we hold
   557 games from two 1150+ teams **on our card-for-card 60**
   (`replays/ntumlnoob_31-07-2026`, 330 g; `replays/sixth_sense_31-07-2026`, 227 g).
3. **True strength is the target, not the displayed number.** The LB reads
   decision-identical agents 63–87 points apart (§8ak) and cannot resolve any
   candidate we own; but scores keep converging through the two continued-play
   weeks after 08-17, and Round 2's BO3 plays the real agent. So "the ladder
   cannot show it" (day-25 item 4) is true today and irrelevant by September.

**Closed axes stay closed** (fifteen + ensembling). Nothing below re-opens
encoding, capacity, data volume, demonstrator cloning, RL fine-tune, search,
sequencing, deck swaps, or the near-tie band.

---

## F1 — Mirror-conditioned disagreement mining (highest ceiling)

**Question:** in MIRROR games specifically, where do the 1150+ pilots
systematically diverge from our clone, and is any divergence a *dominated-option*
class (the only shape rules have ever won on — 3/3 vs 0/4, §3)?

**Why this is not B7 reopened:** B7 *trained on* expert actions and lost −55/−92
(§8t/§8u). F1 *reads* them as an audit target — the method that produced §8f,
§8y, §8ah — and its output is a rule or a priced defect, never a training target.
⛔ It is also not E3: no teacher gets built, and near-tie relabeling is closed
(§8bd). Large-margin confident disagreements are the object, not near-ties.

**Steps, each gating the next:**

1. **Sizing gate (~30 min):** count mirror games and mirror decisions inside the
   two dumps (opponent-archetype match per the `p19`/`PARKED-corpus-coverage`
   slot method). **Gate: ≥100 mirror games**; if under, widen by targeted mining
   of top-band mirror episodes (manifest `avg_score` ≥1100, both seats
   Grimmsnarl) — allowed, this is a named-demonstrator/top-band use, not anchor
   selection.
2. **Disagreement extraction:** run `policy_v5_s2` over the expert mirror
   decisions (reuse `context_accuracy.py` / `p16_policy_disagree.py` machinery,
   `--equiv` so free ties don't count). Keep contexts where the expert's action ≠
   clone top-1 **and the clone's margin is large** — confident disagreement, the
   opposite of the closed near-tie band.
3. **Cluster and size:** group by phase / action type / card; rank by
   frequency × margin. **Per-cluster sizing gate: ≥0.5 firings/game** (the bar
   that killed Morgrem 0.2, Pokégear 0.27, the Archaludon rule 0.187).
4. **Classify with the discriminator:** dominated (provable by arithmetic from
   the board) → rule candidate. Tradeoff → ⛔ no rule (0/4); it becomes report
   material, or a *feature* defect only if priced first the way §8au priced its
   (the B1-instance-5 bar — the encoding axis stays closed otherwise).
5. **Watch 3–5 games** from the top clusters in the visualizer — §8ah's method;
   it beat fifteen days of A/Bs once already.
6. **Any rule candidate A/Bs clean:** byte-identical net both arms, rule
   toggled — the ±25 Elo seed nuisance cancels exactly. Mirror direct n≥2,000 +
   the 7-anchor weighted check.

**Kill criterion (pre-registered):** no cluster passes sizing AND classifies as
dominated → **F1 closes as a chapter** — "what the 1150s do differently in the
mirror is tradeoffs, which is precisely the class a clone cannot be repaired
toward with rules" is a real finding and fits §7b.

**Prediction (register now, check later):** the sizing gate passes easily
(their band is >70% mirror); ≥1 cluster passes the frequency bar; but most
clusters classify as *tradeoffs* and F1 most likely ships nothing while
producing the report's strongest §7b addendum. If a dominated class exists, my
guess is it lives in targeting/bench-management, not attack selection (this
deck has one payable attack).

## F2 — The seed harvest (guaranteed expected value, no new ideas)

**Fact:** the training seed is a ±25 Elo pure nuisance (§8bg: s2 +25.8, s3
−24.4, same everything). We have sampled **four** seeds and shipped the best of
three screened. Order statistics on σ≈25: best-of-12 ≈ +35–40 Elo over the
median seed — comparable to the v4 state block, for zero ideas.

**Protocol (selection bias is the whole danger — pay for it):**

1. Screen `policy_v5_s4` (trained, never screened) vs `policy_v5_s1`, shipped
   config, mirror direct n=1,400 (~12 min). ⚠ Also the day-25 item-5 debt:
   **confirm `s2` on fresh games** — its 0.537 won a screen of three.
2. Train ~6–8 more seeds (`--seed 5..12`, byte-identical recipe). Screen each
   identically. **Screens pick ONE candidate; screens never ship.**
3. **Confirmation on fresh games:** winner vs the incumbent `policy_v5_s2`,
   shipped config, mirror direct, n≥2,800 (±~0.019).
   **Ship bar (pre-registered): point ≥0.53 AND CI excluding 0.50.**
   Below the bar → keep s2, write the null.
4. If shipped: **submit twice** (both slots get fresh draws of the best agent,
   the board takes the max), no later than **08-15**.

**Prediction:** the screen distribution will span ~50 Elo again; the winner's
fresh-game confirmation vs s2 shrinks toward ~0.51–0.53 (s2 is already a
selected +26 seed) and the bar is a coin flip to clear. Either outcome is fine:
the bundle already holds a selected seed, and the distribution itself is §5.6's
closing figure.

## F3 — The corpus-coverage sizing gate (30 min, close the parked lever)

Run the probe exactly as `docs/experiments/embeddings/PARKED-corpus-coverage.md`
specifies: do the replays we already hold contain games against the missing
archetypes (Alakazam 0.15×, Garchomp 0.06×, Archaludon/Lucario 0.00×), or do
those games not exist at the demonstrators' band? **No training happens off the
back of this gate this side of the freeze** — it opens or kills the lever *on
paper*, and the verdict is a report entry either way.

**Prediction:** killed, or opened-but-declined — §8ac says the blind archetypes
are 0-of-47 in games above rating 900, so the matchups the corpus lacks are
vanishing from our field exactly as we climb, and the mirror (what matters) is
*over*-represented 1.92×.

## F4 — DECLINED, recorded so the next session doesn't relitigate

One β for B8 is the only closed axis with an honestly open door ("unfalsified
rather than refuted", §8ao). Declined for the freeze window: two nulls over
20,000 games, the axis closed **on the method**, the hard stop (08-08) is spent,
and a winner could not be integrated safely by 08-15. It stays a report line.

---

## The calendar this buys

| window | work |
|---|---|
| 08-08 | F3 (30 min) · F2 step 1 (s4 screen + s2 fresh-game confirmation) · F1 step 1 (sizing gate) · rule-2 LB reads on `55326513` |
| 08-09 – 08-11 | F1 steps 2–5 · F2 steps 2–3 in the background (screens are ~12 min each) |
| 08-12 – 08-14 | any F1 rule candidate A/B'd; F2 confirmation; ship decision |
| 08-15 | last safe submission — if shipping, **submit twice** |
| 08-16 – 08-17 | freeze; settled pair rides into continued play |

**EVIDENCE entries land the session each item concludes (§8bh onward). Verdicts
stay blank while cells are in flight (the §8i rule).**
