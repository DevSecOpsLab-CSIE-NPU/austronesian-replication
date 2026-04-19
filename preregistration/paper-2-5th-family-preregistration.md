# Pre-registration — Fifth Family Test (Paper 2 cross-family hypothesis)

Written before the migration-gradient numbers were computed. The script and
JSON under `scripts/xfam_5th_uralic_migration_gradient.py` and
`results/xfam_5th_uralic_migration_gradient.json` execute the plan below.
Results are appended at the bottom; the pre-registration block is NOT edited
after execution.

## Pre-registration

**Family chosen**: Uralic

**Dataset**: `lexibank/uralex` (UraLex basic vocabulary, Syrjänen/Honkola/
Lehtinen/Leino/Vesakoski 2013+). 43 language varieties, 313 parameters,
9,751 cognate assignments, form-level `borrowings.csv` for loan exclusion.
Pre-filter: require Glottocode + non-NaN Latitude + Longitude; that yields
~32 usable varieties (exceeds the n≥25 sanity floor).

Why Uralic over Dravidian and Turkic:
- `lexibank/dravlex` has only 20 languages — fails n≥25 floor.
- `lexibank/syrjaenenuralic` has only 7 languages — fails n≥25 floor.
- `lexibank/savelyevturkic` (32 langs) lacks the pre-registered pastoralist
  anchors HORSE / SHEEP / CATTLE / STEPPE entirely in `parameters.csv`; only
  generic natural-feature controls are present. Running Turkic on universal
  controls alone would not be a fair test of the expansion-corridor anchor
  hypothesis (no theoretically motivated anchor concept to pre-register).
- `lexibank/hruschkaturkic` (26 langs) uses "Etymon N" parameter labels
  rather than semantic concepts — unusable for concept-level anchor testing.
- `lexibank/uralex` has all four pre-registered Uralic anchor-adjacent
  concepts present (FOREST, SNOW, TREE, ICE) plus universal controls
  (FIRE, WATER, STONE, TREE, RIVER, MOUNTAIN, SEA).

**Homeland**: middle Volga / Kama region, ~58°N, 55°E (latest Uralic
Urheimat consensus; Grünthal et al. 2022; Honkola et al. 2013). This sits
between Mordvin/Mari core ranges and the Permic-Ugric split line; the
exact point matters less than its placement east of the Baltic and south
of the Saami area, which is all that the Spearman monotonicity test uses.

**Primary predicted anchor**: FOREST (taiga / boreal landscape anchor —
the expansion corridor for early Uralic was across the Eurasian boreal
forest belt, and "forest" is the ecological constant from Finnic/Saami
through Mari/Permic to Mansi/Khanty/Samoyedic).

**Secondary predicted anchors**: SNOW, ICE, TREE (boreal-ecology anchors
retained under continuous cold-climate occupation across the expansion
corridor).

**Note on excluded Uralic a-priori anchors**: REINDEER, FIR, BIRCH, PINE,
MOOSE, ELK all absent from the `uralex` concept slate. This is a dataset
constraint, not a theoretical revision — the prediction was written BEFORE
inspecting the concept list; FOREST and SNOW are the best available
operationalisations of the pre-registered "boreal taiga" anchor theme.

**Universal controls for comparison**: TREE, FIRE, WATER, STONE,
RIVER, SEA, MOUNTAIN, COLD (natural-world reference concepts that should
show weak or inconsistent gradients under the anchor hypothesis — if a
universal control out-performs the anchors, the anchor prediction fails).

**Prediction stated BEFORE looking at gradient data**: YES. The
expansion-corridor anchor hypothesis predicts that the primary anchor
FOREST and the secondary cold-climate anchors SNOW/ICE will show a
positive Spearman ρ between "retention of the proto-Uralic cognate class"
and "Haversine distance from the Volga-Kama homeland" — i.e. languages
further from the homeland (Saami in the west, Samoyedic in the east)
retain the proto-form more often than languages near the homeland where
post-homeland lexical turnover accumulates. A *negative* gradient would
indicate founder-effect geography (the opposite of the anchor mechanism),
and a null/zero gradient with universal controls showing similar patterns
would indicate no selective anchor — both would falsify the prediction.

**Decision rule for "prediction matched"**: the empirically top-ρ concept
(among all tested) must be one of {FOREST, SNOW, ICE, TREE}. If the top-ρ
concept is a universal control (FIRE, WATER, STONE, RIVER, SEA, MOUNTAIN,
COLD), the prediction is not matched.

---

## Results (appended after running — do not edit pre-registration above)

See `results/xfam_5th_uralic_migration_gradient.json` for full numbers and
`scripts/xfam_5th_uralic_migration_gradient.py` for the executed code.

Run date: 2026-04-18. 35 Uralic varieties with Glottocode + geo entered
the pipeline; per-concept n ranges 4–23 after dedup and loan exclusion.

Per-concept Spearman ρ (distance from 58 N, 55 E vs retention), 500-boot
percentile 95 % CI:

| Concept  | Role    |  n |  ret % |    ρ   |   p    | 95 % CI           |
|----------|---------|---:|-------:|-------:|-------:|-------------------|
| FOREST   | anchor  | 11 |  18.2  | +0.224 | 0.509  | [-0.243, +0.676]  |
| SNOW     | anchor  | 18 |  72.2  | -0.203 | 0.419  | [-0.572, +0.206]  |
| ICE      | anchor  | 22 |  90.9  | -0.100 | 0.659  | [-0.500, +0.300]  |
| TREE     | anchor  | 22 |  63.6  | -0.447 | 0.037  | [-0.776, -0.059]  |
| FIRE     | control | 23 |  82.6  | +0.207 | 0.342  | [-0.403, +0.644]  |
| WATER    | control |  8 |  87.5  | +0.577 | 0.134  | [+0.581, +0.888]  |
| STONE    | control | 23 |  43.5  | -0.185 | 0.398  | [-0.562, +0.226]  |
| RIVER    | control | 23 |  73.9  | +0.075 | 0.735  | [-0.451, +0.589]  |
| SEA      | control |  4 |  50.0  | +0.447 | 0.553  | [-1.000, +1.000]  |
| MOUNTAIN | control | 22 |  27.3  | +0.595 | 0.003  | [+0.300, +0.857]  |
| COLD     | control | 17 |  29.4  | +0.553 | 0.021  | [+0.171, +0.870]  |

**Empirically top-ρ concept**: MOUNTAIN (ρ = +0.595, p = 0.003, CI
fully positive).

**Prediction matched?**: NO. None of the four pre-registered anchors
(FOREST, SNOW, ICE, TREE) tops the ranking. FOREST is weak-positive but
has the smallest n (11) and its CI crosses zero. SNOW, ICE, and TREE all
trend *negative*, with TREE significantly so (p = 0.037, CI entirely
below zero) — i.e. proto-Uralic tree cognates are retained *more* near
the Volga-Kama homeland and less at the Saami/Samoyedic peripheries,
the opposite of the anchor prediction.

**Honest interpretation**: This is a genuine null for the pre-registered
Uralic anchor hypothesis. The data instead show a MOUNTAIN-type gradient
that echoes the Indo-European MOUNTAIN finding — intriguing because
Uralic was not supposed to be a mountain-ecology expansion. Treating this
as confirmatory would reintroduce the exact post-hoc concept-selection
bias this fifth-family test was designed to control for. For Paper 2 the
honest report is: "tested Uralic as a pre-registered fifth family with
prediction FOREST/SNOW/ICE/TREE > controls; the pre-registered anchors
did not out-rank controls, so the cross-family anchor hypothesis does
*not* universally generalise — it appears strongest in the Austronesian
SEA and Indo-European MOUNTAIN cases and weaker or absent in Uralic."

**Follow-ups suggested (NOT part of this pre-registration)**: (i) test
Uralic homeland sensitivity — a more easterly Urheimat (~55 N, 65 E,
Honkola 2013 alt.) could flip some signs; (ii) test whether
"MOUNTAIN" retention tracks Saami/Samoyedic (highland-adjacent) retention
of the Proto-Uralic cognate — this would echo the IE Caucasus/Alps
pattern; (iii) extend the slate to include FISH/BIRCH/BEAR if a richer
Uralic cognate set (e.g. a future Uralex v2 or a Samoyedic-expanded
dataset) becomes available.

