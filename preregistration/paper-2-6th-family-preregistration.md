# Pre-registration — Sixth Family Test (Paper 2 cross-family hypothesis)

Written on 2026-04-19 before the migration-gradient numbers are computed.
The script and JSON under `scripts/xfam_6th_turkic_migration_gradient.py`
and `results/xfam_6th_turkic_migration_gradient.json` execute the plan
below. Results are appended at the bottom; the pre-registration block is
NOT edited after execution.

GitHub issue: #86.

## Pre-registration — Sixth Family Test (Issue #86)

**Date**: 2026-04-19

**Family chosen**: Turkic

**Dataset**: `lexibank/savelyevturkic` (Savelyev and Robbeets 2020,
"The internal structure of the Turkic language family"). 32 Turkic
varieties, 254 Swadesh-style parameters, 8 360 forms, 8 360 cognate
assignments produced by Savelyev's expert classification. After filtering
on Glottocode + non-NaN Latitude + Longitude, **24 varieties** remain
(one below the n>=25 floor adopted in the fifth-family Uralic
pre-registration — accepted as the best available Turkic option; see
"Dataset survey" below).

### Dataset survey and family choice justification

- `lexibank/dravlex` (Kolipakam et al. 2018): only **20** Dravidian
  varieties (below the n>=25 floor) AND none of the pre-registered
  Dravidian anchors (RICE, COCONUT, COTTON) are in the parameter slate;
  only basic-Swadesh ecological controls (TREE, FIRE, WATER, STONE,
  MOUNTAIN) are present. **Rejected**: fails both the size and the
  anchor-coverage criteria.
- `lexibank/hruschkaturkic` (Hruschka et al. 2015): 26 languages with
  geo, but `parameters.csv` uses "Etymon N" labels rather than semantic
  concepts — Concepticon_Gloss is not populated, so concept-level
  anchor testing is impossible. **Rejected**: not concept-addressable.
- `lexibank/savelyevturkic` (Savelyev and Robbeets 2020): 24 varieties
  with geo; full Concepticon mapping. None of the pre-registered Turkic
  anchors (HORSE, SHEEP, STEPPE, FELT, CATTLE, COW, GOAT, WOLF, CAMEL,
  MILK, WHEAT, BARLEY) are in the 254-concept basic-Swadesh slate — only
  DOG and MEAT among animal/subsistence terms. Universal controls SEA,
  TREE, FIRE, WATER, STONE, LEFT, RIGHT, RIVER, SNOW are all present.
  **Accepted**: best available Turkic CLDF with concept-level semantics,
  despite the anchor-coverage gap.
- Mayan: no `lexibank/*mayan*` or Concepticon-mapped Mayan CLDF dataset
  of adequate size exists on GitHub at the time of writing
  (queried 2026-04-19). **Untestable**: no dataset.

Turkic via `savelyevturkic` is therefore chosen as the sixth family,
with the explicit caveat that it lacks pre-registered pastoralist anchor
concepts. This forces the test to be a **universal-controls-only**
replication plus whatever generic-ecology terms (TREE, SEA) serve as
weak proxies for a steppe-expansion hypothesis. The honest purpose of
the sixth test is therefore: *does even a universal-control slate
produce a MOUNTAIN-type gradient in Turkic — or does Turkic, like Uralic,
produce a null for the pre-registered anchor hypothesis?*

**Homeland**: Altai / north-Mongolian steppe, **48.0 N, 100.0 E**
(Savelyev and Robbeets 2020's reconstructed Proto-Turkic Urheimat; also
consistent with Dybo 2007 and Janhunen 1996). This places the homeland
east of the Kazakh steppe expansion corridor and north-west of the
Bulgharic branch separation, with modern Turkic varieties spreading west
(Anatolia, Crimea) and north/east (Yakutia, Sakha). The Spearman
monotonicity test cares about the rank order of distances, not the
exact coordinate.

**Primary predicted anchors**: HORSE, SHEEP, STEPPE, FELT
(pastoralist/nomadic anchors — the classic Proto-Turkic subsistence
package). **All four are absent from `savelyevturkic/parameters.csv`**
(confirmed by exact-match query against Concepticon_Gloss). This is a
dataset-coverage gap, not a theoretical revision.

**Secondary predicted anchors** (best available proxies in the
concept slate): **DOG** (domesticated animal, present), **MEAT**
(pastoralist subsistence, present), **MOUNTAIN OR HILL** (tested in
the Concepticon_Gloss form "MOUNTAIN OR HILL" — Altai highlands as
launch-geography proxy), and **HORN (ANATOMY)** (livestock-body
part, often retained in pastoralist lexicons). These are registered
*before* running as the pre-registered fallback anchor list for the
Turkic test.

**Universal controls for comparison** (present in the concept slate):
TREE, FIRE, WATER, STONE, LEFT, RIGHT, RIVER, SEA, SNOW, COLD.
These should show weak or inconsistent gradients under the anchor
hypothesis — if a universal control out-performs every fallback anchor,
the Turkic prediction fails.

**Prediction stated BEFORE looking at gradient data**: YES.

1. The steppe-expansion anchor hypothesis predicts that the fallback
   pastoralist anchors DOG and MEAT will show positive Spearman rho
   between "retention of the proto-Turkic cognate class" and "Haversine
   distance from the Altai homeland" — i.e. languages further from
   the homeland retain the proto-form more often than languages near
   the homeland where post-homeland lexical turnover accumulates.
2. A *null* result (no anchor or control clearly positive) would
   replicate the Uralic null and strengthen the reading that the
   cross-family anchor signal is strongest in Austronesian SEA and
   weakest or absent in continental expansion families.
3. A *negative* gradient would be a founder-effect signature (opposite
   of the anchor mechanism) and would falsify the anchor prediction.

**Decision rule for "prediction matched"** (binding):

- **supported**: the empirically top-rho concept (among all concepts
  tested) is one of {DOG, MEAT, MOUNTAIN OR HILL, HORN (ANATOMY)}
  AND its bootstrap 95% CI is disjoint from the second-best concept's
  CI.
- **partial**: the top-rho concept is a fallback anchor but its CI
  overlaps the next concept's CI.
- **registered_null**: the top-rho concept is a universal control.
- **untestable**: fewer than three concepts produce a defined Spearman
  rho (e.g. n<3 after dedup).

The pre-registered anchor list (HORSE, SHEEP, STEPPE, FELT) is **not
removed from the verdict logic**: if any of the four were somehow
present under a renamed gloss (checked at script startup), they would
take precedence over the fallback anchors in the decision rule.

**Method** (identical to the fifth-family Uralic pipeline):

1. Load forms.csv, cognates.csv, languages.csv, parameters.csv.
2. Exclude forms flagged as loans via the `Loan` column in forms.csv
   (savelyevturkic has no separate `borrowings.csv`; the form-level
   Loan boolean is the documented loan channel for this dataset).
3. For each concept, join forms to cognates on Form_ID.
4. Deduplicate by Glottocode via majority-vote of Cognateset_ID.
5. Retention = 1 iff the language's majority-vote Cognateset_ID equals
   the global modal Cognateset_ID for the concept (proto-form proxy).
6. Haversine distance from the pre-registered homeland (48.0 N, 100.0 E).
7. Spearman rho(distance, retention).
8. 500-sample bootstrap percentile 95% CI, `np.random.default_rng(42)`.

Seed, bootstrap count, dedup rule, retention definition, loan exclusion
policy, distance metric, and decision rule are all fixed by this
pre-registration.

**Written**: BEFORE running gradient computation.

## Pre-registration signed off
[No edits past this line. Results appended below.]

---

## Actual results

Run date: 2026-04-19. See `results/xfam_6th_turkic_migration_gradient.json`
for full numbers and `scripts/xfam_6th_turkic_migration_gradient.py` for
the executed code.

**Languages entering the pipeline**: 24 Turkic varieties with Glottocode
+ non-NaN Latitude + Longitude (one below the n>=25 floor; this gap was
recorded in the pre-registration above, not introduced post-hoc).

**Pre-registered primary-anchor availability**: HORSE, SHEEP, STEPPE,
FELT — **all four absent** from savelyevturkic's concept slate, as
predicted in the pre-registration.

**Per-concept Spearman ρ** (distance from 48 N, 100 E vs retention),
500-boot percentile 95% CI:

| Concept          | Role             |  n | ret % |    ρ    |   p    | 95 % CI            |
|------------------|------------------|---:|------:|--------:|-------:|--------------------|
| HORSE            | anchor_primary   |  0 |   —   |   —     |   —    | absent             |
| SHEEP            | anchor_primary   |  0 |   —   |   —     |   —    | absent             |
| STEPPE           | anchor_primary   |  0 |   —   |   —     |   —    | absent             |
| FELT             | anchor_primary   |  0 |   —   |   —     |   —    | absent             |
| DOG              | anchor_fallback  | 24 |  83.3 | −0.032  | 0.881  | [−0.576, +0.511]   |
| MEAT             | anchor_fallback  | 21 | 100.0 |  n/a    |  n/a   | 100% retention     |
| MOUNTAIN OR HILL | anchor_fallback  | 24 |  83.3 | +0.000  | 1.000  | [−0.448, +0.411]   |
| HORN (ANATOMY)   | anchor_fallback  | 22 | 100.0 |  n/a    |  n/a   | 100% retention     |
| TREE             | control          | 24 |  75.0 | −0.445  | 0.029  | [−0.738, +0.015]   |
| FIRE             | control          | 22 | 100.0 |  n/a    |  n/a   | 100% retention     |
| WATER            | control          | 24 | 100.0 |  n/a    |  n/a   | 100% retention     |
| STONE            | control          | 24 | 100.0 |  n/a    |  n/a   | 100% retention     |
| LEFT             | control          | 18 |  83.3 | −0.158  | 0.531  | [−0.566, +0.274]   |
| RIGHT            | control          | 24 |  75.0 | −0.361  | 0.083  | [−0.624, +0.000]   |
| RIVER            | control          | 22 |  45.5 | −0.389  | 0.074  | [−0.725, +0.026]   |
| SEA              | control          | 15 | 100.0 |  n/a    |  n/a   | 100% retention     |
| SNOW             | control          | 24 | 100.0 |  n/a    |  n/a   | 100% retention     |
| COLD             | control          | 24 |  87.5 | +0.191  | 0.371  | [−0.044, +0.447]   |

**Empirically top-ρ concept**: COLD (universal control), ρ = +0.191,
p = 0.371, CI = [−0.044, +0.447] (crosses zero).

**Verdict (by pre-registered decision rule)**: **registered_null**.
The top positive-ρ concept is a universal control (COLD), not a
pre-registered anchor. The CI of the top concept is also not disjoint
from zero, so even the weak positive signal is not robust. Of the three
fallback anchors that yielded a defined ρ (DOG, MOUNTAIN OR HILL, plus
the effectively-null MEAT/HORN at 100% retention), none produced a
clearly positive gradient.

**On the 100%-retention concepts**: seven of 18 tested concepts (MEAT,
HORN (ANATOMY), FIRE, WATER, STONE, SEA, SNOW) are retained as a single
proto-Turkic cognate class across every sampled variety. This is
expected for a tightly-related language family with a relatively recent
(c. 500 CE) common ancestor and an expert-classified cognate set — the
Spearman monotonicity test is structurally uninformative when there is
zero retention variance. It is not a pipeline bug.

**On the TREE and RIGHT/RIVER negative gradients**: TREE shows
ρ = −0.445 (p = 0.029), i.e. proto-Turkic TREE cognates are retained
*more* near the Altai homeland and *less* at the Anatolian/Sakha
peripheries — the opposite of the anchor prediction. RIGHT and RIVER
trend negative at the 10% level. This is consistent with a founder-
effect / dialect-continuum pattern where peripheral varieties accrete
innovations; it is the inverse of the anchor mechanism and does not
support the sixth-family anchor hypothesis.

**Honest interpretation for Paper 2**: This is a genuine **registered
null** for the pre-registered Turkic anchor hypothesis. Combined with
the Uralic null (fifth family), the cross-family anchor signal now
pattern-matches as: strong in Austronesian (SEA), Indo-European
(MOUNTAIN), and the two other positive families of the original five;
null in Uralic; null in Turkic. The honest report for Paper 2 is that
the anchor hypothesis does **not** universally generalise across
expansion families, and the sixth-family test registered as a pre-
registered replication reinforces — rather than overturns — this
pattern. Two consecutive pre-registered nulls after three successes
(AN, IE, Bantu — assuming ST ambiguous) supports a narrower claim:
anchor gradients are a conditional rather than universal feature of
large language-family expansions.

**Data-coverage caveats** (documented above in the pre-registration):
(i) n=24 is one below the n>=25 floor; (ii) all four primary
pastoralist anchors (HORSE, SHEEP, STEPPE, FELT) are absent from the
Savelyev & Robbeets 2020 concept slate, so the test fell back to
domesticated-animal / landscape proxies; (iii) seven concepts reach
100% retention and contribute no gradient signal. A richer Turkic
dataset that includes HORSE/SHEEP (e.g. a future lexibank expansion
of Dybo 2007 or an ASJP-level pastoralist-term extraction) would
sharpen this test, but does not exist in CLDF form today.

