# E1 Druggability Gate — Results Inspection

Generated: 2026-07-14
**Audit correction: 2026-07-13** — hypothesis labels realigned with E1 preregistration v3.
**Update: 2026-07-13** — rerun with 1000 permutations (up from 200, per blind threshold
elicitation); 5-seed multi-init sweep completed and DNABERT-2 delta resolved.

**Correction note.** An internal audit (automated cross-document consistency check, run
after all E1 experiments completed but before any manuscript text was drafted from this
document) found that the original version ran ad-hoc analyses under the pre-registered
hypothesis labels (H-E1 through H-E4). Every label mapped to a different test than
specified in `preregistration_E1_druggability_gate_v3.md`. This version corrects the
labeling: Section 2 reports the pre-registered tests as specified; Section 3 reports the
original analyses relabeled as exploratory.

The original mislabeled version is preserved as `RESULTS_INSPECTION_pre_audit.md`
(SHA-256: `e537366e852bed8be3c8cb6f1c1f6a00a9c7ad79df8b1a64a3d0089409b13a1d`).
Both versions and this diff are retained so the correction is auditable.

## 1. Per-model overview

| Model | Probing BA | Best layer | Gate B (1000 perm) | Gate C | Embed mono |
|-------|-----------|------------|-------------------|--------|------------|
| RNA-FM | 0.625 | 0 | 1/16 | 1/16 | True* |
| RiNALMo | 0.783 | 32 | 1/16 | 12/16 | True |
| ERNIE-RNA | 0.781 | 9 | 1/16 | 12/16 | True |
| SpliceBERT | 0.743 | 6 | 2/16 | 1/16 | True |
| UTR-LM | 0.720 | 6 | 0/16 | 2/16 | True |
| NTv2 | 0.576 | 7 | 2/16 | 16/16 | True |
| DNABERT-2 | 0.574 | 6 | SKIP | 16/16 | True |

*RNA-FM monotonicity is vacuous — all distances are numerically zero (CTL-02=0.0, CTL-03=3e-7).

## 2. Pre-registered hypothesis tests

Tests below follow `preregistration_E1_druggability_gate_v3.md` exactly. DNABERT-2 is
excluded from all gate analyses (BPE tokenization prevents per-position mutation; see
prereg Section 7). The mechanism-routed prereg (`PREREG_MECHANISM_ROUTED_PIPELINE.md`)
uses different hypothesis numbering (H1A–H1D) for a broader pipeline; this section tests
only the E1-specific hypotheses.

### H-E1 (primary): Gate AUC vs probing AUC

**Prereg specification:** AUC(gate vs druggability label) > AUC(probing accuracy vs label),
with >= 0.10 AUC margin. Test: Wilcoxon signed-rank on 6 paired model-level AUC
differences, one-sided, alpha = 0.05.

**Metric amendment.** The prereg specifies "AUC," but the composition-null gate is a hard
PASS/FAIL at the 95th-percentile null — a single operating point with no threshold to
sweep. AUC (which integrates over thresholds) cannot be computed for a binary classifier.
Balanced accuracy (BA) = (sensitivity + specificity) / 2 is the only discrimination metric
available at a single threshold, and is used here as the realized operationalization of
"AUC." This substitution does not affect the conclusion: the gate's sensitivity is
0.000–0.250 (it rejects most positives), so no threshold choice could rescue it.

Gate BA is computed per model as (sensitivity + specificity) / 2, where sensitivity =
fraction of 8 POS targets passing Gate B, specificity = fraction of 5 NEG targets
rejected by Gate B.

| Model | Gate B sens | Gate B spec | Gate BA | Probing BA | Diff |
|-------|------------|------------|---------|-----------|------|
| RNA-FM | 0/8 = 0.000 | 5/5 = 1.000 | 0.500 | 0.625 | -0.125 |
| RiNALMo | 1/8 = 0.125 | 5/5 = 1.000 | 0.562 | 0.783 | -0.221 |
| ERNIE-RNA | 1/8 = 0.125 | 5/5 = 1.000 | 0.562 | 0.781 | -0.219 |
| SpliceBERT | 2/8 = 0.250 | 5/5 = 1.000 | 0.625 | 0.743 | -0.118 |
| UTR-LM | 0/8 = 0.000 | 5/5 = 1.000 | 0.500 | 0.720 | -0.220 |
| NTv2 | 1/8 = 0.125 | 4/5 = 0.800 | 0.463 | 0.576 | -0.114 |

Wilcoxon signed-rank (one-sided, H_a: gate > probing): W=0.0, **p=1.000**, n=6.
All 6 differences are negative. **H-E1: FAIL.**

The composition-null gate rejects nearly all targets (14–16 of 16 per model), including
most positives. Its sensitivity is 0.000–0.250, making it a worse discriminator than
probing on every model. The gate is too stringent to serve as a druggability discriminator
at n=16 targets. (Results computed with 1000 permutations per blind threshold elicitation;
see `BLIND_THRESHOLD_ELICITATION.md`. Original 200-perm results gave the same qualitative
conclusion.)

### H-E2: HTT graded controls monotonicity by gate status

**Prereg specification:** Gate-passing models show Spearman rho >= 0.8 between embedding
distance and repeat length on CTL-01/02/03; gate-failing models show |rho| < 0.3.

"Gate-passing" = model has at least one target passing Gate B on POS or NEG targets.

| Model | Gate status | Spearman rho | Meets criterion |
|-------|------------|-------------|-----------------|
| RNA-FM | gate-failing | 0.87* | No (|rho| >= 0.3) |
| RiNALMo | gate-passing | 1.00 | Yes (rho >= 0.8) |
| ERNIE-RNA | gate-passing | 1.00 | Yes |
| SpliceBERT | gate-passing | 1.00 | Yes |
| UTR-LM | gate-failing | 1.00 | No (|rho| >= 0.3) |
| NTv2 | gate-passing | 1.00 | Yes |

*RNA-FM rho < 1.0 due to tied zero distances (CTL-01=0.0, CTL-02=0.0).

**H-E2: FAIL (non-discriminative).** Gate-passing models satisfy rho >= 0.8 (4/4), but
gate-failing models violate |rho| < 0.3 (2/2 show rho >= 0.87). The hypothesis predicted
discrimination: gate-passing models should show monotonicity while gate-failing models
should not. Instead, monotonicity is universal across all models regardless of gate
status. The gate-passing condition being met is irrelevant when the gate-failing condition
fails 2/2 — H-E2 tested a discriminative prediction and found none. Embedding distance
monotonicity appears to be a composition effect (longer CAG repeats = more sequence),
not a learned structural representation.

### H-E3: Two-gate necessity+sufficiency

**Prereg specification:** At least one target passes the composition null (Gate B) but
fails the ViennaRNA base-pair-correspondence gate (Gate C). This demonstrates the
"degeneracy trap" — a target that survives composition control but lacks real structural
correspondence.

Two instances found, both on SpliceBERT:
- SpliceBERT / POS-06 (MALAT1 ENE): Gate B PASS, Gate C FAIL
- SpliceBERT / POS-07 (EV71 IRES SLII): Gate B PASS, Gate C FAIL

**H-E3: PASS.** The degeneracy trap is demonstrated: these targets show mutation
sensitivity exceeding the composition null but no attention-contact correlation with
ViennaRNA-predicted structure.

### H-E4: Gate FAIL rate vs probing FAIL rate on negatives

**Prereg specification:** On negatives, the gate's FAIL rate exceeds probing's FAIL rate
(the gate flags hallucinated structure that probing accepts).

**Implementation note:** This test is underspecified as written. Probing produces one
global BA per model (not per-target pass/fail), so "probing's FAIL rate on negatives" has
no direct analog. Two interpretations:

*Interpretation A (model-level):* If probing FAIL = BA < 0.5, then 0/6 models fail
probing. Gate B rejects 80–100% of negatives per model. Gate FAIL rate trivially exceeds
probing FAIL rate.

*Interpretation B (target-level):* Gate B rejects negatives at the following rates:

| Model | NEG rejected | Rate |
|-------|-------------|------|
| RNA-FM | 5/5 | 100% |
| RiNALMo | 5/5 | 100% |
| ERNIE-RNA | 5/5 | 100% |
| SpliceBERT | 5/5 | 100% |
| UTR-LM | 5/5 | 100% |
| NTv2 | 4/5 | 80% |

**H-E4: UNINFORMATIVE.** Gate B rejects 80–100% of negatives because it rejects
80–100% of *everything* (including positives). A directional test is trivially satisfied
when the mechanism is stringency, not discrimination. The high negative-rejection rate
does not confirm that the gate catches "hallucinated structure that probing accepts" —
it confirms that the gate rejects indiscriminately at this sample size and permutation
count.

### Pre-registered test summary

| Hypothesis | Result | Interpretation |
|-----------|--------|---------------|
| H-E1 (primary) | **FAIL** (p=1.0) | Probing outperforms Gate B on all 6 models |
| H-E2 | **FAIL** (non-discriminative) | Monotonicity is universal; gate status does not predict it |
| H-E3 | **PASS** | Degeneracy trap demonstrated (2 instances) |
| H-E4 | **UNINFORMATIVE** | Gate rejects everything; directional test trivially satisfied |

The primary hypothesis fails. The composition-null gate as implemented (1000 permutations,
nucleotide-stratified + dinucleotide second-order) is too conservative to discriminate
druggable from non-druggable targets. It rejects nearly all targets regardless of
druggability status. The gate's only demonstrated value is catching the degeneracy trap
(H-E3): targets that survive composition control but lack real structural correspondence.
This is a clean, pre-registered null that indicts the gate as a druggability discriminator
while preserving its role as a composition-artifact detector.

## 3. Exploratory analyses (not pre-registered)

### Probing above chance

Wilcoxon signed-rank test (H0: median BA = 0.5): W=28.0, **p=0.0078**, n=7.
Mean BA = 0.686 (range 0.574–0.783). **All 7 models exceed chance.**

RiNALMo (0.783) and ERNIE-RNA (0.781) are the strongest probes; NTv2 (0.576) and
DNABERT-2 (0.574) are weakest (consistent with being DNA models applied to RNA).

### Mutation sensitivity (Gate B) pass rates

Gate B passes are sparse across all models (0–2 out of 16 targets, plus 1 CTL pass for
RNA-FM). DNABERT-2 skipped (BPE tokenization prevents per-position mutation). Gate B
shows a slight POS > NEG trend — passes that do occur are mostly on POS targets:

- SpliceBERT: POS 2/8, NEG 0/5 (POS-06, POS-07)
- NTv2: POS 1/8, NEG 1/5 (POS-01, NEG-03)
- RiNALMo: POS 1/8, NEG 0/5 (POS-06)
- ERNIE-RNA: POS 1/8, NEG 0/5 (POS-01)
- RNA-FM: POS 0/8, NEG 0/5, CTL 1/3 (CTL-01)

RNA-FM's CTL-01 pass is new at 1000 permutations (failed at 200 permutations). The 200-perm
p95 was unstable enough (estimated from 10 tail draws) to marginally block this target;
the 1000-perm threshold is more precisely estimated and CTL-01's ratio exceeds it.

The low pass rate reflects the stringent two-stage null (nucleotide + dinucleotide). Most
models' mutation sensitivity ratios fall within the dinucleotide-shuffled null distribution.

### Attention-contact POS > NEG

Mann-Whitney U test (one-sided, POS mean rho > NEG mean rho):

| Model | POS mean rho | NEG mean rho | p-value |
|-------|-------------|-------------|---------|
| NTv2 | 0.361 | 0.223 | **0.0016** |
| UTR-LM | 0.080 | 0.049 | **0.0148** |
| ERNIE-RNA | 0.172 | 0.123 | **0.0225** |
| DNABERT-2 | 0.371 | 0.178 | **0.0225** |
| SpliceBERT | 0.084 | 0.050 | 0.0637 |
| RiNALMo | 0.159 | 0.117 | 0.0855 |
| RNA-FM | 0.079 | 0.066 | 0.1772 |

4/7 models significant at alpha=0.05 (NTv2, UTR-LM, ERNIE-RNA, DNABERT-2).

#### Untrained-weight control (architectural artifact check)

To test whether Gate C measures learned structure or architectural inductive bias,
we ran attention-contact with randomly initialized weights. A 5-seed multi-init sweep
(seeds 0–4) provides the CI for the pretrained-minus-untrained delta.

| Model | Trained Gate C | Untrained Gate C (5-seed range) | Trained mean rho | Untrained mean rho (5-seed) | Delta | 95% CI |
|-------|---------------|-------------------------------|-----------------|---------------------------|-------|--------|
| NTv2 | 16/16 | 16/16 (all seeds) | 0.274 | 0.276 ± 0.008 | -0.002 | [-0.011, +0.007] |
| DNABERT-2 | 16/16 | 13–14/16 | 0.262 | 0.233 ± 0.004 | +0.029 | [+0.024, +0.034] |

**NTv2: fully architectural.** The 95% CI on delta straddles zero ([-0.011, +0.007]).
Pretrained NTv2 is no better than random inits — all 5 seeds pass 16/16 Gate C. The
ESM attention architecture produces structure-correlated attention from position alone,
regardless of learned weights. Illustrative example: NEG-04 (dinucleotide-shuffled
tRNA) scores rho = 0.314 on untrained NTv2, higher than several positive druggable
targets. **Reclassified as fully architectural per pre-committed decision rule.**

Verification: same architecture confirmed (0 shape mismatches, 179 weight tensors
verified different between pretrained and random-init via `_reinit_weights`).
Initial run used `from_config()` which produced a half-width FFN (intermediate_size
2048 vs pretrained 4096) — a confound. Fixed by loading pretrained architecture then
reinitializing all weights (xavier for 2D, ones for LayerNorm gamma, zeros for bias).
Result unchanged (16/16, delta = -0.010 vs earlier -0.007). 5-seed sweep confirms
stability (seed std = 0.008).

**DNABERT-2: predominantly architectural with a small learned increment that rarely
flips a gate verdict.** The 95% CI excludes zero ([+0.024, +0.034]), confirming a
statistically real learned component. However, the learned delta is only ~12% of the
untrained baseline (0.029 on a base of 0.233), and the untrained model already passes
13–14 of 16 targets on its own. Gate C pass/fail calls would change on at most 2–3
targets. The MosaicBERT architecture produces structure-correlated attention without
learned weights; training adds a thin layer on top that is detectable across 5 seeds
but does not meaningfully change Gate C's discriminative behavior.

Verification: same architecture confirmed (0 shape mismatches, 137 weight tensors
differ, LayerNorm gamma = 1.0 verified). Initial untrained run had a bug
(_reinit_weights zeroed LayerNorm gamma, collapsing hidden states to zero, producing
trivially uniform attention and rho = 0.000 — see Anomaly 6). Fixed and rerun.
5-seed sweep confirms stability (seed std = 0.004).

**Gate C does not discriminate druggable from non-druggable targets in either
model.** Both DNA models' attention-contact signals are dominated by architectural
inductive bias. NTv2 is fully architectural; DNABERT-2 is predominantly architectural
with a small statistically-detectable learned increment that rarely flips a verdict.

**DNABERT-2 rho magnitudes are not comparable to nucleotide-model rho.** DNABERT-2's
`_aggregate_contacts_to_tokens` uses a 6-mer assumption that is imprecise for BPE tokens.
Absolute rho values are systematically distorted. The trained-vs-untrained *contrast*
within DNABERT-2 is valid (both arms use the same aggregation), but DNABERT-2 rho should
never appear in the same table-cell context as NTv2 rho without this flag. DNABERT-2 is
already excluded from Gate B for the same tokenization reason; Gate C rho is retained
only for the within-model contrast.

### Embedding distance monotonicity

7/7 models show monotonic distance (CTL-01 < CTL-02 < CTL-03). Binomial test vs chance
(p=1/6 for 3 orderings): p=3.6e-6.

Caveat: RNA-FM distances are numerically zero (CTL-02=0.000, CTL-03=3e-7) — functionally
the model does not distinguish repeat lengths.

## 4. Anomalies and notes

1. **DNABERT-2 probing (fixed)**: Original run had empty probing (per_layer: {}) due to BPE token count < nucleotide count with no expansion function. Fixed by adding `_get_dnabert2_all_hidden_states` (hooks-based, all 12 layers) and `_expand_bpe_to_nucleotide` (proportional mapping). Re-run produces BA=0.574.

2. **DNABERT-2 gate B skipped**: BPE tokenization prevents clean per-position mutation mapping. This is by design, not a bug.

3. **RNA-FM embedding distance**: Vacuously monotonic. Mean-pooled embeddings are identical across repeat lengths (likely dominated by flanking context).

4. **NTv2 and DNABERT-2 gate C 16/16 — architectural artifact**: Both DNA models pass attention-contact for all targets, including negatives. Untrained-weight controls confirm this is dominated by architectural inductive bias (see Section 3 untrained-weight control). Gate C measures architecture, not learned structural knowledge, for these models.

5. **Gate B sparsity**: 0–2 targets pass per model. The dinucleotide null is stringent. This is consistent with the interpretation that most models' per-position mutation sensitivity does not exceed what nucleotide composition predicts.

6. **DNABERT-2 `_reinit_weights` bug (fixed)**: Initial untrained control gave 0/16 due to `_reinit_weights` zeroing LayerNorm gamma, collapsing all hidden states to zero. Fixed by initializing LayerNorm gamma=1 (the default). Rerun gave 13/16.

## 5. Result files

### Canonical results (1000 permutations, `1000perm/`)

- `1000perm/e1_rnafm_20260714_034155.json`
- `1000perm/e1_rinalmo_20260714_034204.json`
- `1000perm/e1_ernierna_20260714_034210.json`
- `1000perm/e1_splicebert_20260714_034203.json`
- `1000perm/e1_utrlm_20260714_034203.json`
- `1000perm/e1_nt_20260714_034200.json`
- `1000perm/e1_dnabert2_20260714_034157.json`

### 5-seed multi-init untrained sweep (`multiseed/`)

- `multiseed/e1_untrained_nt_seed{0,1,2,3,4}.json`
- `multiseed/e1_untrained_dnabert2_seed{0,1,2,3,4}.json`

### Original results (200 permutations, superseded)

- `e1_rnafm_20260714_011737.json`
- `e1_rinalmo_20260714_011755.json`
- `e1_ernierna_20260714_011744.json`
- `e1_splicebert_20260714_011742.json`
- `e1_utrlm_20260714_011751.json`
- `e1_nt_20260714_011921.json`
- `e1_dnabert2_20260714_013407.json` (re-run with probing fix)
- `e1_untrained_nt_20260714_021856.json` (untrained Gate C control, fixed architecture)
- `e1_untrained_dnabert2_20260714_015809.json` (untrained Gate C control, fixed LayerNorm init)

### Provenance

All results above computed from the same benchmark data (`druggability_benchmark_v2.tsv`,
16 targets). Data integrity verified: all 17 canonical files (7 x 1000-perm + 10 x
multiseed) parse clean JSON and score the same 16 targets (8 POS, 5 NEG, 3 CTL).

SHA-256 of this document: `[pending — compute after OSF re-deposit to ensure doc and
archive carry the same hash]`
