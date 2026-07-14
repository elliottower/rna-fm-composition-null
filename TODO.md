# TODO

## Target: ~300 stratified targets, not 4,000

The 16-target panel constrains statistical power and invites instant reviewer rejection. But 4,000 unstructured Rfam families is diminishing returns. The sweet spot is ~300 structured targets — enough to stratify by RNA type, length, and GC content with real error bars per stratum.

- ~100-150: kills "n is absurd", gives composition-null rejection rate a tight CI
- ~300-500: enables per-stratum survival rates — the paper becomes "a finding about the field"
- Beyond ~500: diminishing returns unless making a modeling claim about the whole RNA universe

The original 16 curated therapeutic targets fold in as a labeled druggability stratum within the larger panel.

## Three must-do upgrades (matter as much as N)

### 1. Experimental structure ground truth for a subset

ViennaRNA-only ground truth will get pushed at every RNA venue. For 30-50 targets, use experimental structure: SHAPE-MaP/DMS-MaPseq where available, or Rfam covariation structures for conserved families. Don't need it for all 300 — need it to show the finding holds on gold-standard structure.

### 2. Untrained-weight control on all 7 models, not just 2

Currently NTv2 + DNABERT-2 only. The architectural-artifact result is the strongest finding and deserves full coverage. Running the 5-seed sweep on all 7 models is one extra pass per model per seed — trivial compute, closes the biggest hole.

### 3. A positive control stratum that survives

Scariest reviewer question: "does anything pass your gate, or is the null just miscalibrated?" Need a stratum designed to survive if genuine structure exists (highly conserved riboswitches with strong covariation evidence). If they pass and lncRNAs fail, that's a result. If nothing passes, the positive control shows the null isn't broken.

## Reframe the primary hypothesis

At ~300 label-free targets, "gate predicts druggability" can't be primary. Make primary: survival-rate-by-stratum (how pervasive is the composition confound across RNA types). Keep druggability discrimination as secondary on the curated therapeutic subset.

## Venue strategy

- PLOS Comp Biol Methods track (primary) — controls-as-method + pre-registered negative is exactly their lane
- NAR / NAR G&B (RNA-native alternative) — strong if experimental structure subset is included
- Genome Biology (reach) — only after all three upgrades land; two-week desk-reject lottery ticket

## Framing decision (not yet committed)

Two options:
- "How pervasive is the composition confound across RNA types" (survey, PLOS/NAR-GAB friendly)
- "No current RNA FM encodes structure beyond composition" (modeling claim, higher ceiling but needs positive control to be airtight)
