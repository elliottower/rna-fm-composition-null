"""E1 druggability gate experiment — composition-null gate vs probing on 16 targets.

Pre-registered: preregistration_E1_druggability_gate_v3.md
Benchmark: druggability_benchmark_v2.tsv

Runs three metrics per model x target:
  Metric A: logistic-regression probing (balanced accuracy, GroupKFold by target)
  Metric B: mutation-sensitivity ratio R under composition-null (gate)
  Metric C: attention-contact correlation (Spearman, best head, max-over-layers)

Usage:
    modal run --detach batch1/code/modal_e1_druggability.py::run_rnafm_e1
    modal run --detach batch1/code/modal_e1_druggability.py::run_rinalmo_e1
    modal run --detach batch1/code/modal_e1_druggability.py::run_ernierna_e1
    modal run --detach batch1/code/modal_e1_druggability.py::run_splicebert_e1
    modal run --detach batch1/code/modal_e1_druggability.py::run_utrlm_e1
    modal run --detach batch1/code/modal_e1_druggability.py::run_ntv2_e1
    modal run --detach batch1/code/modal_e1_druggability.py::run_dnabert2_e1
"""
from pathlib import Path

import modal

app = modal.App("e1-druggability-gate")

SIBLING_REPOS = Path(__file__).resolve().parent.parent.parent.parent
RNAFM_WEIGHTS_LOCAL = SIBLING_REPOS / "causal-rna" / "pretrained" / "pytorch_model.bin"
CODE_DIR = Path(__file__).resolve().parent
BENCHMARK_TSV_LOCAL = Path(__file__).resolve().parent.parent / "data" / "druggability_benchmark_v2.tsv"
BENCHMARK_TSV_CONTAINER = "/root/druggability_benchmark_v2.tsv"

output_vol = modal.Volume.from_name("e1-druggability-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install(
        "torch==2.13.0",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-learn==1.9.0",
        "tqdm==4.68.4",
        "matplotlib==3.11.0",
        "transformers==4.57.6",
        "einops==0.8.2",
        "requests==2.34.2",
        "ViennaRNA==2.7.0",
    )
    .add_local_dir(str(CODE_DIR / "lib"), "/root/project/lib")

    .add_local_file(str(RNAFM_WEIGHTS_LOCAL), "/root/pretrained/pytorch_model.bin")
    .add_local_file(str(BENCHMARK_TSV_LOCAL), BENCHMARK_TSV_CONTAINER)
)

multimolecule_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install(
        "torch==2.13.0",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-learn==1.9.0",
        "tqdm==4.68.4",
        "matplotlib==3.11.0",
        "transformers==5.13.1",
        "multimolecule==0.2.0",
        "ViennaRNA==2.7.0",
    )
    .add_local_dir(str(CODE_DIR / "lib"), "/root/project/lib")

    .add_local_file(str(BENCHMARK_TSV_LOCAL), BENCHMARK_TSV_CONTAINER)
)

nt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install(
        "torch==2.13.0",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-learn==1.9.0",
        "tqdm==4.68.4",
        "matplotlib==3.11.0",
        "transformers==4.57.6",
        "einops==0.8.2",
        "ViennaRNA==2.7.0",
    )
    .add_local_dir(str(CODE_DIR / "lib"), "/root/project/lib")

    .add_local_file(str(BENCHMARK_TSV_LOCAL), BENCHMARK_TSV_CONTAINER)
)

dnabert2_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.0.1",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "scikit-learn==1.5.2",
        "tqdm==4.68.4",
        "matplotlib==3.9.3",
        "transformers==4.28.0",
        "einops==0.8.2",
        "requests==2.34.2",
        "ViennaRNA==2.7.0",
    )
    .run_commands("pip uninstall -y triton || true")
    .add_local_dir(str(CODE_DIR / "lib"), "/root/project/lib")

    .add_local_file(str(BENCHMARK_TSV_LOCAL), BENCHMARK_TSV_CONTAINER)
)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK LOADER
# ═══════════════════════════════════════════════════════════════════════════════

# HTT construct flanks (from NM_002111.7, verified in Paper C)
HTT_5PRIME_FLANK = "AUGAAGGCCUUCGAGUCCCUCAAGUCCUUC"
HTT_3PRIME_FLANK = (
    "CAACAGCCGCCACCGCCGCCGCCGCCGCCGCCGCCUCCUCAGCUUCCUCAG"
    "CCGCCGCCGCAGGCACAGCCGCUGCUGCCUCAGCCGCAGCCGCCCCCGCCG"
    "CCGCCCCCGCCGCCACCCGGCCCGGCUGUGGCUGAGGAGCCGCUGCACCGAC"
)

HTT_REPEAT_COUNTS = {"CTL-01": 17, "CTL-02": 36, "CTL-03": 60}


def _dinucleotide_shuffle(seq, rng):
    """Dinucleotide-preserving shuffle via Altschul-Erickson Euler path algorithm.

    Preserves both mononucleotide AND dinucleotide composition by constructing
    a random Eulerian path through the dinucleotide graph of the input sequence.
    This is the same algorithm implemented in uShuffle (Jiang et al. 2008).
    """
    from collections import defaultdict

    edges = defaultdict(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])

    for nuc in edges:
        rng.shuffle(edges[nuc])

    path = [seq[0]]
    stack = [seq[0]]

    while stack:
        v = stack[-1]
        if edges[v]:
            u = edges[v].pop()
            stack.append(u)
        else:
            path.append(stack.pop())

    path.reverse()
    result = "".join(path[:-1])

    if len(result) != len(seq):
        raise ValueError(
            f"Euler path length {len(result)} != input length {len(seq)}; "
            f"dinucleotide graph may not be Eulerian"
        )

    return result


def _generate_runtime_sequences(rng_seed=42):
    """Generate NEG-04 (shuffled tRNA) and NEG-05 (random GC-matched) at runtime."""
    import numpy as np

    rng = np.random.default_rng(rng_seed)

    # NEG-04: dinucleotide-preserved shuffle of tRNA-Phe (from PDB 1EHZ)
    # Uses Altschul-Erickson Euler path algorithm (same as uShuffle)
    trna_phe = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"
    neg04 = _dinucleotide_shuffle(trna_phe, rng)

    # NEG-05: random sequence GC-matched to mean positive target GC content (~55%)
    gc_frac = 0.55
    n = 76
    bases = []
    for _ in range(n):
        if rng.random() < gc_frac:
            bases.append(rng.choice(["G", "C"]))
        else:
            bases.append(rng.choice(["A", "U"]))
    neg05 = "".join(bases)

    return neg04, neg05


def _fold_pinned(seq):
    """Fold with explicitly pinned ViennaRNA parameters.

    Uses fold_compound with per-call model details, not the global RNA.fold()
    which reads mutable global state. Parameters:
      temperature=37.0, dangles=2, noLP=0 (lonely pairs permitted),
      energy parameters=Turner 2004 (ViennaRNA 2.7.0 default).
    """
    import RNA
    md = RNA.md()
    md.temperature = 37.0
    md.dangles = 2
    md.noLP = 0
    fc = RNA.fold_compound(seq, md)
    structure, mfe = fc.mfe()
    return structure, mfe


def load_benchmark():
    """Load benchmark TSV and generate runtime sequences + ViennaRNA folds.

    ViennaRNA fold parameters are explicitly pinned per-call via fold_compound
    (not RNA.fold() which uses mutable global state). See _fold_pinned().
    """
    local_path = Path(__file__).parent.parent / "data" / "druggability_benchmark_v2.tsv"
    benchmark_path = local_path if local_path.exists() else Path(BENCHMARK_TSV_CONTAINER)
    targets = []

    with open(benchmark_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            if not line.strip():
                continue
            fields = line.strip().split("\t")
            row = dict(zip(header, fields))
            targets.append(row)

    neg04_seq, neg05_seq = _generate_runtime_sequences(rng_seed=42)

    for t in targets:
        tid = t["Target_ID"]

        # Fill runtime-generated sequences
        if tid == "NEG-04":
            t["Sequence_5to3"] = neg04_seq
        elif tid == "NEG-05":
            t["Sequence_5to3"] = neg05_seq

        # Fill HTT constructs
        if tid in HTT_REPEAT_COUNTS:
            n = HTT_REPEAT_COUNTS[tid]
            t["Sequence_5to3"] = HTT_5PRIME_FLANK + "CAG" * n + HTT_3PRIME_FLANK

        # Fold with explicitly pinned parameters (not global RNA.fold())
        seq = t["Sequence_5to3"]
        structure, mfe = _fold_pinned(seq)
        t["Dot_Bracket"] = structure
        t["mfe"] = mfe

    return targets


def targets_to_rna_dicts(targets):
    """Convert benchmark rows to the dict format expected by analysis functions."""
    rna_list = []
    for t in targets:
        rna_list.append({
            "name": t["Target_ID"],
            "family": t["RNA_Type"],
            "rfam_acc": "NA",
            "sequence": t["Sequence_5to3"],
            "structure": t["Dot_Bracket"],
            "source": t.get("Source", ""),
            "label": t.get("Label", ""),
            "class": t.get("Class", ""),
        })
    return rna_list


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS (adapted from modal_expanded_rfam.py / modal_htt_experiments.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_paired_and_unpaired_positions(structure, max_pos=None):
    paired = []
    unpaired = []
    stack = []
    n = len(structure) if max_pos is None else min(len(structure), max_pos)
    for i in range(n):
        c = structure[i]
        if c == '(':
            stack.append(i)
            paired.append(i)
        elif c == ')':
            if stack:
                stack.pop()
            paired.append(i)
        else:
            unpaired.append(i)
    return paired, unpaired


def _parse_structure_to_contacts(structure):
    import numpy as np
    n = len(structure)
    contacts = np.zeros((n, n), dtype=np.float32)
    stack = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                contacts[i, j] = 1.0
                contacts[j, i] = 1.0
    return contacts


def _expand_6mer_to_nucleotide(embeddings, n_nucleotides):
    import numpy as np
    n_tokens, d = embeddings.shape
    expanded = np.empty((n_nucleotides, d), dtype=embeddings.dtype)
    for i in range(n_nucleotides):
        token_idx = min(i // 6, n_tokens - 1)
        expanded[i] = embeddings[token_idx]
    return expanded


def _aggregate_contacts_to_tokens(contact_map, n_nucleotides, n_tokens):
    import numpy as np
    token_contacts = np.zeros((n_tokens, n_tokens), dtype=np.float32)
    for ti in range(n_tokens):
        si = ti * 6
        ei = min(si + 6, n_nucleotides)
        for tj in range(n_tokens):
            sj = tj * 6
            ej = min(sj + 6, n_nucleotides)
            if contact_map[si:ei, sj:ej].any():
                token_contacts[ti, tj] = 1.0
    return token_contacts


def _validate_structure(rna):
    seq = rna["sequence"]
    structure = rna["structure"]
    n = min(len(seq), len(structure))
    if n < 20:
        return False, "too short"
    paired, unpaired = _get_paired_and_unpaired_positions(structure, max_pos=n)
    if len(paired) < 5 or len(unpaired) < 5:
        return False, f"insufficient positions (stem={len(paired)}, loop={len(unpaired)})"
    return True, "ok"


def _get_dnabert2_all_hidden_states(model, tokens):
    """Extract per-layer hidden states from MosaicBERT via hooks.

    MosaicBERT ignores output_hidden_states, so we hook each encoder layer.
    Returns list of tensors, each (seq_len_no_special, d_model).
    """
    import torch

    hidden_states = []
    hooks = []

    for layer in model.encoder.layer:
        def _hook(module, input, output, _list=hidden_states):
            out = output[0] if isinstance(output, tuple) else output
            _list.append(out.detach())
        hooks.append(layer.register_forward_hook(_hook))

    with torch.no_grad():
        model(tokens)

    for h in hooks:
        h.remove()

    result = []
    for hs in hidden_states:
        if hs.dim() == 2:
            hs = hs.unsqueeze(0)
        result.append(hs[0, 1:-1, :])
    return result


def _expand_bpe_to_nucleotide(embeddings, n_nucleotides):
    """Map BPE token embeddings to per-nucleotide embeddings (proportional)."""
    import numpy as np
    n_tokens, d = embeddings.shape
    expanded = np.empty((n_nucleotides, d), dtype=embeddings.dtype)
    for i in range(n_nucleotides):
        token_idx = min(int(i * n_tokens / n_nucleotides), n_tokens - 1)
        expanded[i] = embeddings[token_idx]
    return expanded


def _extract_dnabert2_attention(model, tokens):
    import torch
    import math

    qkv_outputs = []
    hooks = []

    for layer in model.encoder.layer:
        self_attn = layer.attention.self

        def _hook_wqkv(module, input, output, _list=qkv_outputs):
            _list.append(output.detach())

        if hasattr(self_attn, 'Wqkv'):
            hooks.append(self_attn.Wqkv.register_forward_hook(_hook_wqkv))
        else:
            for h in hooks:
                h.remove()
            return None

    with torch.no_grad():
        model(tokens)

    for h in hooks:
        h.remove()

    if not qkv_outputs:
        return None

    attentions = []
    n_heads = 12
    for qkv in qkv_outputs:
        if qkv.dim() == 2:
            qkv = qkv.unsqueeze(0)
        batch, seq_len, three_d = qkv.shape
        d_model = three_d // 3
        d_head = d_model // n_heads
        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, n_heads, d_head).transpose(1, 2)
        k = k.view(batch, seq_len, n_heads, d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
        attn_weights = torch.softmax(scores, dim=-1)
        attentions.append(attn_weights)

    return attentions


def run_mutation_sensitivity(adapter, model_key, rna_structures, device="cuda",
                             n_permutations=1000):
    """Mutation sensitivity with composition-null (Metric B).

    1000 permutations, 95th percentile threshold, max-over-layers null.
    Dinucleotide-stratified second-order null on first-order survivors.
    """
    import numpy as np
    from scipy.spatial.distance import cosine
    from tqdm import tqdm

    results = {"model": model_key, "metric": "mutation_sensitivity_gate",
                "n_permutations": n_permutations, "per_target": {}}
    COMPLEMENT = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C', 'T': 'A'}
    all_ratios = []

    for rna_idx, rna in enumerate(tqdm(rna_structures, desc=f"  {model_key} gate")):
        valid, reason = _validate_structure(rna)
        if not valid:
            results["per_target"][rna["name"]] = {"skipped": reason}
            continue

        seq = list(rna["sequence"])
        structure = rna["structure"]
        n_pos = min(len(seq), len(structure))

        paired, unpaired = _get_paired_and_unpaired_positions(structure, max_pos=n_pos)
        if len(paired) < 3 or len(unpaired) < 3:
            results["per_target"][rna["name"]] = {"skipped": "insufficient positions"}
            continue

        wt_seq = "".join(seq[:n_pos])
        if model_key in ("rnafm", "rinalmo", "utrlm", "ernierna", "splicebert"):
            tokens = adapter.tokenize(wt_seq.replace("T", "U")).to(device)
        else:
            tokens = adapter.tokenize(wt_seq).to(device)
        wt_embs = adapter.get_all_layer_embeddings(tokens)
        n_layers = len(wt_embs)

        labels = np.array(['stem' if i in paired else 'loop' for i in range(n_pos)])

        per_layer_dists = {l: np.zeros(n_pos) for l in range(n_layers)}
        per_layer_valid = {l: np.zeros(n_pos, dtype=bool) for l in range(n_layers)}

        for pos in range(n_pos):
            nuc = wt_seq[pos]
            comp = COMPLEMENT.get(nuc)
            if comp is None or comp == nuc:
                continue

            mut_seq = list(wt_seq)
            mut_seq[pos] = comp
            mut_seq_str = "".join(mut_seq)

            if model_key in ("rnafm", "rinalmo", "utrlm", "ernierna", "splicebert"):
                mut_tokens = adapter.tokenize(mut_seq_str.replace("T", "U")).to(device)
            else:
                mut_tokens = adapter.tokenize(mut_seq_str).to(device)
            mut_embs = adapter.get_all_layer_embeddings(mut_tokens)

            for layer_idx in range(n_layers):
                wt_layer = wt_embs[layer_idx].cpu().numpy()
                mut_layer = mut_embs[layer_idx].cpu().numpy()
                if model_key == "nt":
                    wt_layer = _expand_6mer_to_nucleotide(wt_layer, n_pos)
                    mut_layer = _expand_6mer_to_nucleotide(mut_layer, n_pos)
                if pos >= min(wt_layer.shape[0], mut_layer.shape[0]):
                    continue
                dist = cosine(wt_layer[pos], mut_layer[pos])
                per_layer_dists[layer_idx][pos] = float(dist)
                per_layer_valid[layer_idx][pos] = True

        real_ratios_per_layer = {}
        for layer_idx in range(n_layers):
            dists = per_layer_dists[layer_idx]
            valid_mask = per_layer_valid[layer_idx]
            stem_mask = np.array([labels[i] == 'stem' and valid_mask[i] for i in range(n_pos)])
            loop_mask = np.array([labels[i] == 'loop' and valid_mask[i] for i in range(n_pos)])
            if int(stem_mask.sum()) < 3 or int(loop_mask.sum()) < 3:
                continue
            mean_stem = float(np.mean(dists[stem_mask]))
            mean_loop = float(np.mean(dists[loop_mask]))
            ratio = mean_stem / mean_loop if mean_loop > 1e-10 else 0.0
            real_ratios_per_layer[layer_idx] = {
                "ratio": ratio, "mean_stem": mean_stem, "mean_loop": mean_loop,
                "n_stem": int(stem_mask.sum()), "n_loop": int(loop_mask.sum()),
            }

        if not real_ratios_per_layer:
            results["per_target"][rna["name"]] = {"skipped": "no valid layers"}
            continue

        best_layer_idx = max(real_ratios_per_layer, key=lambda l: real_ratios_per_layer[l]["ratio"])
        best_ratio = real_ratios_per_layer[best_layer_idx]["ratio"]

        # Nucleotide-stratified null (max-over-layers)
        valid_layers = sorted(real_ratios_per_layer.keys())
        perm_rng = np.random.default_rng(42 + rna_idx * 1000)
        nucs_full = np.array(list(wt_seq[:n_pos]))

        null_max_ratios = []
        for _ in range(n_permutations):
            shuffled_labels = labels.copy()
            for nuc in ('A', 'U', 'G', 'C'):
                nuc_idx = np.where(nucs_full == nuc)[0]
                if len(nuc_idx) < 2:
                    continue
                shuffled_labels[nuc_idx] = perm_rng.permutation(shuffled_labels[nuc_idx])

            max_perm_ratio = 0.0
            for layer_idx in valid_layers:
                dists = per_layer_dists[layer_idx]
                valid_mask = per_layer_valid[layer_idx]
                vi = np.where(valid_mask)[0]
                vd = dists[vi]
                vs = shuffled_labels[vi]
                perm_stem = vd[vs == 'stem']
                perm_loop = vd[vs == 'loop']
                if len(perm_stem) == 0 or len(perm_loop) == 0:
                    continue
                ml = np.mean(perm_loop)
                max_perm_ratio = max(max_perm_ratio, np.mean(perm_stem) / ml if ml > 1e-10 else 0.0)
            null_max_ratios.append(max_perm_ratio)

        nuc_null_95th = float(np.percentile(null_max_ratios, 95)) if null_max_ratios else 0.0
        exceeds_nuc = best_ratio > nuc_null_95th

        rna_result = {
            "best_ratio": best_ratio, "best_layer": int(best_layer_idx),
            "nuc_null_95th": nuc_null_95th, "exceeds_nuc_null": exceeds_nuc,
            "n_stem": real_ratios_per_layer[best_layer_idx]["n_stem"],
            "n_loop": real_ratios_per_layer[best_layer_idx]["n_loop"],
        }

        # Dinucleotide null on first-order survivors
        if exceeds_nuc:
            dinucs_full = np.array([
                ('N' if i == 0 else wt_seq[i - 1]) + wt_seq[i] for i in range(n_pos)
            ])
            dinuc_null_max_ratios = []
            for _ in range(n_permutations):
                shuffled_labels = labels.copy()
                for dinuc in np.unique(dinucs_full):
                    dinuc_idx = np.where(dinucs_full == dinuc)[0]
                    if len(dinuc_idx) < 2:
                        continue
                    shuffled_labels[dinuc_idx] = perm_rng.permutation(shuffled_labels[dinuc_idx])
                max_perm_ratio = 0.0
                for layer_idx in valid_layers:
                    dists = per_layer_dists[layer_idx]
                    valid_mask = per_layer_valid[layer_idx]
                    vi = np.where(valid_mask)[0]
                    vd = dists[vi]
                    vs = shuffled_labels[vi]
                    perm_stem = vd[vs == 'stem']
                    perm_loop = vd[vs == 'loop']
                    if len(perm_stem) == 0 or len(perm_loop) == 0:
                        continue
                    ml = np.mean(perm_loop)
                    max_perm_ratio = max(max_perm_ratio, np.mean(perm_stem) / ml if ml > 1e-10 else 0.0)
                dinuc_null_max_ratios.append(max_perm_ratio)

            dinuc_null_95th = float(np.percentile(dinuc_null_max_ratios, 95))
            rna_result["dinuc_null_95th"] = dinuc_null_95th
            rna_result["exceeds_dinuc_null"] = best_ratio > dinuc_null_95th
            rna_result["gate_pass"] = best_ratio > dinuc_null_95th
        else:
            rna_result["gate_pass"] = False

        results["per_target"][rna["name"]] = rna_result
        all_ratios.append(best_ratio)

    results["mean_best_ratio"] = float(np.mean(all_ratios)) if all_ratios else 0.0
    n_pass = sum(1 for r in results["per_target"].values() if r.get("gate_pass"))
    results["n_gate_pass"] = n_pass
    results["n_targets_scored"] = len(all_ratios)
    return results


def run_attention_contact(adapter, model_key, rna_structures, device="cuda"):
    """Attention-contact correlation (Metric C)."""
    import numpy as np
    from scipy import stats
    from tqdm import tqdm

    results = {"model": model_key, "metric": "attention_contact", "per_target": {}}

    has_attention = model_key in ("rnafm", "nt", "rinalmo", "utrlm", "ernierna", "splicebert", "dnabert2")
    if not has_attention:
        results["skipped"] = "no attention"
        return results

    for rna in tqdm(rna_structures, desc=f"  {model_key} attention"):
        valid, reason = _validate_structure(rna)
        if not valid:
            results["per_target"][rna["name"]] = {"skipped": reason}
            continue

        seq = rna["sequence"]
        structure = rna["structure"]
        n_pos = min(len(seq), len(structure))

        if model_key in ("rnafm", "rinalmo", "utrlm", "ernierna", "splicebert"):
            tokens = adapter.tokenize(seq[:n_pos].replace("T", "U")).to(device)
        else:
            tokens = adapter.tokenize(seq[:n_pos]).to(device)

        import torch
        with torch.no_grad():
            if model_key == "dnabert2":
                attentions = _extract_dnabert2_attention(adapter.model, tokens)
            else:
                out = adapter.model(tokens, output_attentions=True, return_dict=True)
                attentions = getattr(out, 'attentions', None) if not isinstance(out, tuple) else (out[3] if len(out) > 3 else None)

        if attentions is None:
            results["per_target"][rna["name"]] = {"skipped": "no attention returned"}
            continue

        contact_map = _parse_structure_to_contacts(structure)
        contact_flat = contact_map[np.triu_indices(n_pos, k=1)]

        best_corr = 0.0
        best_layer = 0

        for layer_idx, attn_tensor in enumerate(attentions):
            attn = attn_tensor[0].cpu().numpy()
            n_heads = attn.shape[0]

            eff_n = n_pos
            eff_contact_flat = contact_flat
            if model_key in ("rnafm", "rinalmo", "utrlm", "ernierna", "splicebert"):
                attn = attn[:, 1:n_pos+1, 1:n_pos+1]
            elif model_key in ("nt", "dnabert2"):
                attn = attn[:, 1:-1, 1:-1]
                n_tok = attn.shape[1]
                tok_contact = _aggregate_contacts_to_tokens(contact_map, n_pos, n_tok)
                eff_n = n_tok
                eff_contact_flat = tok_contact[np.triu_indices(n_tok, k=1)]

            if attn.shape[1] != eff_n or attn.shape[2] != eff_n:
                continue

            for head in range(n_heads):
                attn_sym = (attn[head] + attn[head].T) / 2
                attn_flat = attn_sym[np.triu_indices(eff_n, k=1)]
                if np.std(attn_flat) < 1e-10:
                    continue
                rho, _ = stats.spearmanr(attn_flat, eff_contact_flat)
                if np.isnan(rho):
                    rho = 0.0
                if rho > best_corr:
                    best_corr = float(rho)
                    best_layer = layer_idx

        gate_c_pass = best_corr >= 0.10
        results["per_target"][rna["name"]] = {
            "best_corr": best_corr, "best_layer": best_layer, "gate_c_pass": gate_c_pass,
        }

    return results


def run_structure_probing(adapter, model_key, rna_structures, device="cuda"):
    """Structure probing (Metric A)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from tqdm import tqdm

    results = {"model": model_key, "metric": "structure_probing", "per_layer": {}}

    all_embeddings = []
    all_labels = []
    all_groups = []

    for rna_idx, rna in enumerate(tqdm(rna_structures, desc=f"  {model_key} probing")):
        valid, reason = _validate_structure(rna)
        if not valid:
            continue

        seq = rna["sequence"]
        structure = rna["structure"]
        n_pos = min(len(seq), len(structure))

        if model_key in ("rnafm", "rinalmo", "utrlm", "ernierna", "splicebert"):
            tokens = adapter.tokenize(seq[:n_pos].replace("T", "U")).to(device)
        else:
            tokens = adapter.tokenize(seq[:n_pos]).to(device)

        if model_key == "dnabert2":
            layer_embs = _get_dnabert2_all_hidden_states(adapter.model, tokens)
        else:
            layer_embs = adapter.get_all_layer_embeddings(tokens)
        paired, unpaired = _get_paired_and_unpaired_positions(structure, max_pos=n_pos)
        lab = np.zeros(n_pos, dtype=np.int32)
        for p in paired:
            lab[p] = 1
        groups = np.full(n_pos, rna_idx, dtype=np.int32)

        for layer_idx in range(len(layer_embs)):
            emb = layer_embs[layer_idx].cpu().numpy()
            if emb.shape[0] < n_pos:
                if model_key == "nt":
                    emb = _expand_6mer_to_nucleotide(emb, n_pos)
                elif model_key == "dnabert2":
                    emb = _expand_bpe_to_nucleotide(emb, n_pos)
                else:
                    continue
            emb_trimmed = emb[:n_pos]
            if layer_idx >= len(all_embeddings):
                all_embeddings.append([])
                all_labels.append([])
                all_groups.append([])
            all_embeddings[layer_idx].append(emb_trimmed)
            all_labels[layer_idx].append(lab)
            all_groups[layer_idx].append(groups)

    for layer_idx in range(len(all_embeddings)):
        X = np.concatenate(all_embeddings[layer_idx], axis=0)
        y = np.concatenate(all_labels[layer_idx], axis=0)
        g = np.concatenate(all_groups[layer_idx], axis=0)

        if len(np.unique(y)) < 2:
            continue
        n_groups = len(np.unique(g))
        n_splits = min(5, n_groups)
        if n_splits < 2:
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0)
        gkf = GroupKFold(n_splits=n_splits)
        scores = cross_val_score(clf, X, y, cv=gkf, groups=g, scoring="balanced_accuracy")
        results["per_layer"][str(layer_idx)] = {
            "mean_balanced_accuracy": float(np.mean(scores)),
            "std_balanced_accuracy": float(np.std(scores)),
            "n_samples": int(len(y)),
            "n_targets": n_groups,
        }

    if results["per_layer"]:
        best = max(results["per_layer"].items(), key=lambda x: x[1]["mean_balanced_accuracy"])
        results["best_layer"] = int(best[0])
        results["best_accuracy"] = best[1]["mean_balanced_accuracy"]

    return results


def run_embedding_distance(adapter, model_key, rna_structures, device="cuda"):
    """Embedding distance from CAG17 wild-type for HTT graded controls (H-E2)."""
    import numpy as np
    from scipy.spatial.distance import cosine

    ctl_targets = [r for r in rna_structures if r["class"] == "CONTROL_GRADED"]
    if len(ctl_targets) < 2:
        return {"model": model_key, "metric": "embedding_distance", "skipped": "no graded controls"}

    results = {"model": model_key, "metric": "embedding_distance", "distances_from_wt": []}

    embeddings = {}
    for rna in ctl_targets:
        seq = rna["sequence"]
        if model_key in ("rnafm", "rinalmo", "utrlm", "ernierna", "splicebert"):
            tokens = adapter.tokenize(seq.replace("T", "U")).to(device)
        else:
            tokens = adapter.tokenize(seq).to(device)
        layer_embs = adapter.get_all_layer_embeddings(tokens)
        best_layer = layer_embs[-1].cpu().numpy()
        embeddings[rna["name"]] = best_layer.mean(axis=0)

    wt_name = "CTL-01"
    if wt_name not in embeddings:
        return {"model": model_key, "metric": "embedding_distance", "skipped": "no CTL-01"}

    wt_emb = embeddings[wt_name]
    for name, emb in sorted(embeddings.items()):
        d = 0.0 if name == wt_name else float(cosine(wt_emb, emb))
        results["distances_from_wt"].append({"target": name, "distance": d})

    dists = [x["distance"] for x in results["distances_from_wt"] if x["target"] != wt_name]
    if len(dists) >= 2:
        results["monotonic"] = all(dists[i] <= dists[i+1] for i in range(len(dists)-1))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def _save_result(result, filename):
    import json
    out_dir = Path("/output/e1_druggability")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    output_vol.commit()
    print(f"Saved {out_path}")


def _run_full_pipeline(adapter, model_key, targets, rna_list, device="cuda"):
    """Run all three metrics and save combined result."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Metric B: gate (mutation sensitivity with composition null)
    # DNABERT-2 cannot run mutation sensitivity (BPE tokenization)
    if model_key != "dnabert2":
        r_gate = run_mutation_sensitivity(adapter, model_key, rna_list, device=device)
        print(f"  Gate: {r_gate['n_gate_pass']}/{r_gate['n_targets_scored']} pass")
    else:
        r_gate = {"model": model_key, "metric": "mutation_sensitivity_gate",
                  "skipped": "BPE tokenization prevents per-position mutation sensitivity"}

    # Metric C: attention-contact
    r_attn = run_attention_contact(adapter, model_key, rna_list, device=device)

    # Metric A: probing
    r_probe = run_structure_probing(adapter, model_key, rna_list, device=device)
    print(f"  Probing: best={r_probe.get('best_accuracy', 0):.3f}")

    # H-E2: embedding distance on graded controls
    r_embed = run_embedding_distance(adapter, model_key, rna_list, device=device)

    # Fold metadata
    fold_info = {}
    for t in targets:
        fold_info[t["Target_ID"]] = {
            "sequence_length": len(t["Sequence_5to3"]),
            "structure_length": len(t["Dot_Bracket"]),
            "mfe": t.get("mfe"),
            "n_paired": t["Dot_Bracket"].count("("),
            "n_unpaired": t["Dot_Bracket"].count("."),
        }

    import torch
    result = {
        "model": model_key,
        "timestamp": ts,
        "torch_version": torch.__version__,
        "n_targets": len(targets),
        "viennarna_version": "2.7.0",
        "viennarna_params": "fold_compound with md: temperature=37.0, dangles=2, noLP=0, Turner2004",
        "gate": r_gate,
        "attention_contact": r_attn,
        "probing": r_probe,
        "embedding_distance": r_embed,
        "fold_info": fold_info,
    }
    _save_result(result, f"e1_{model_key}_{ts}.json")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PER-MODEL MODAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(image=image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_rnafm_e1():
    import sys
    from datetime import datetime
    sys.path.insert(0, "/root/project")
    from lib.adapters.rnafm import RNAFMAdapter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] RNA-FM E1 druggability experiment starting")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)
    print(f"  Loaded {len(targets)} targets, all folded with ViennaRNA 2.7.0")

    adapter = RNAFMAdapter(weight_path="/root/pretrained/pytorch_model.bin")
    adapter.load(pretrained=True)
    adapter.model = adapter.model.to("cuda")

    _run_full_pipeline(adapter, "rnafm", targets, rna_list)
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] RNA-FM E1 complete")


@app.function(image=multimolecule_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_rinalmo_e1():
    import sys
    from datetime import datetime
    sys.path.insert(0, "/root/project")
    from lib.adapters.rinalmo import RiNALMoAdapter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] RiNALMo E1 druggability experiment starting")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)

    adapter = RiNALMoAdapter()
    adapter.load(pretrained=True)
    adapter.model = adapter.model.to("cuda")

    _run_full_pipeline(adapter, "rinalmo", targets, rna_list)
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] RiNALMo E1 complete")


@app.function(image=multimolecule_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_ernierna_e1():
    import sys
    from datetime import datetime
    sys.path.insert(0, "/root/project")
    from lib.adapters.ernierna import ERNIERNAAdapter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] ERNIE-RNA E1 druggability experiment starting")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)

    adapter = ERNIERNAAdapter()
    adapter.load(pretrained=True)
    adapter.model = adapter.model.to("cuda")

    _run_full_pipeline(adapter, "ernierna", targets, rna_list)
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] ERNIE-RNA E1 complete")


@app.function(image=multimolecule_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_splicebert_e1():
    import sys
    from datetime import datetime
    sys.path.insert(0, "/root/project")
    from lib.adapters.splicebert import SpliceBERTAdapter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] SpliceBERT E1 druggability experiment starting")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)

    adapter = SpliceBERTAdapter()
    adapter.load(pretrained=True)
    adapter.model = adapter.model.to("cuda")

    _run_full_pipeline(adapter, "splicebert", targets, rna_list)
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] SpliceBERT E1 complete")


@app.function(image=multimolecule_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_utrlm_e1():
    import sys
    from datetime import datetime
    sys.path.insert(0, "/root/project")
    from lib.adapters.utrlm import UTRLMAdapter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] UTR-LM E1 druggability experiment starting")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)

    adapter = UTRLMAdapter()
    adapter.load(pretrained=True)
    adapter.model = adapter.model.to("cuda")

    _run_full_pipeline(adapter, "utrlm", targets, rna_list)
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] UTR-LM E1 complete")


@app.function(image=nt_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_ntv2_e1():
    import sys
    from datetime import datetime
    sys.path.insert(0, "/root/project")
    from lib.adapters.nt import NTAdapter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] NT v2 E1 druggability experiment starting")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)

    adapter = NTAdapter()
    adapter.load(pretrained=True)
    adapter.model = adapter.model.to("cuda")

    _run_full_pipeline(adapter, "nt", targets, rna_list)
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] NT v2 E1 complete")


@app.function(image=dnabert2_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_dnabert2_e1():
    import sys
    from datetime import datetime
    sys.path.insert(0, "/root/project")
    from lib.adapters.dnabert2 import DNABERT2Adapter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] DNABERT-2 E1 druggability experiment starting")
    print("  NOTE: DNABERT-2 scored on attention-contact + probing only (no mutation sensitivity)")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)

    adapter = DNABERT2Adapter()
    adapter.load(pretrained=True)
    adapter.model = adapter.model.to("cuda")

    _run_full_pipeline(adapter, "dnabert2", targets, rna_list)
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] DNABERT-2 E1 complete")


# ═══════════════════════════════════════════════════════════════════════════════
# UNTRAINED-WEIGHT CONTROLS (Gate C architectural artifact check)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_untrained_attention_control(adapter, model_key, targets, rna_list, device="cuda"):
    """Run attention-contact only with untrained weights. Saves as separate result."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    r_attn = run_attention_contact(adapter, model_key, rna_list, device=device)

    n_pass = sum(1 for v in r_attn.get("per_target", {}).values()
                 if isinstance(v, dict) and v.get("gate_c_pass"))
    n_scored = len([v for v in r_attn.get("per_target", {}).values()
                    if isinstance(v, dict) and "gate_c_pass" in v])
    print(f"  Untrained Gate C: {n_pass}/{n_scored} pass")

    import torch
    result = {
        "model": model_key,
        "condition": "untrained_random_weights",
        "timestamp": ts,
        "torch_version": torch.__version__,
        "n_targets": len(targets),
        "attention_contact": r_attn,
    }
    _save_result(result, f"e1_untrained_{model_key}_{ts}.json")


@app.function(image=nt_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_ntv2_untrained_gate_c():
    import sys
    from datetime import datetime
    sys.path.insert(0, "/root/project")
    from lib.adapters.nt import NTAdapter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] NTv2 UNTRAINED Gate C control starting")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)

    adapter = NTAdapter()
    adapter.load(pretrained=False)
    adapter.model = adapter.model.to("cuda")

    _run_untrained_attention_control(adapter, "nt", targets, rna_list)
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] NTv2 untrained control complete")


@app.function(image=dnabert2_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_dnabert2_untrained_gate_c():
    import sys
    from datetime import datetime
    sys.path.insert(0, "/root/project")
    from lib.adapters.dnabert2 import DNABERT2Adapter

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] DNABERT-2 UNTRAINED Gate C control starting")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)

    adapter = DNABERT2Adapter()
    adapter.load(pretrained=False)
    adapter.model = adapter.model.to("cuda")

    _run_untrained_attention_control(adapter, "dnabert2", targets, rna_list)
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] DNABERT-2 untrained control complete")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-SEED UNTRAINED SWEEP (robustness check on Δ)
# Decision rule: if 95% CI on mean Δ crosses zero → "fully architectural"
# ═══════════════════════════════════════════════════════════════════════════════

SEEDS = [0, 1, 2, 3, 4]


def _run_seeded_untrained(adapter_cls, model_key, seed, image_ref):
    """Run one seed of the untrained attention-contact control."""
    import torch
    torch.manual_seed(seed)

    adapter = adapter_cls()
    adapter.load(pretrained=False)
    adapter.model = adapter.model.to("cuda")

    targets = load_benchmark()
    rna_list = targets_to_rna_dicts(targets)

    r_attn = run_attention_contact(adapter, model_key, rna_list, device="cuda")

    n_pass = sum(1 for v in r_attn.get("per_target", {}).values()
                 if isinstance(v, dict) and v.get("gate_c_pass"))
    n_scored = len([v for v in r_attn.get("per_target", {}).values()
                    if isinstance(v, dict) and "gate_c_pass" in v])
    print(f"  {model_key} seed={seed}: {n_pass}/{n_scored} pass")

    result = {
        "model": model_key,
        "condition": "untrained_random_weights",
        "seed": seed,
        "attention_contact": r_attn,
    }
    _save_result(result, f"e1_untrained_{model_key}_seed{seed}.json")
    return result


@app.function(image=nt_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_ntv2_untrained_seed(seed: int):
    import sys
    sys.path.insert(0, "/root/project")
    from lib.adapters.nt import NTAdapter
    print(f"NTv2 untrained seed={seed}")
    _run_seeded_untrained(NTAdapter, "nt", seed, None)


@app.function(image=dnabert2_image, gpu="A10G", timeout=86400, volumes={"/output": output_vol})
def run_dnabert2_untrained_seed(seed: int):
    import sys
    sys.path.insert(0, "/root/project")
    from lib.adapters.dnabert2 import DNABERT2Adapter
    print(f"DNABERT-2 untrained seed={seed}")
    _run_seeded_untrained(DNABERT2Adapter, "dnabert2", seed, None)


@app.local_entrypoint()
def run_multiseed_sweep():
    """Launch 5-seed untrained sweep for both NTv2 and DNABERT-2."""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Launching 5-seed untrained sweep: 2 models x 5 seeds")
    futures = []
    for seed in SEEDS:
        futures.append(run_ntv2_untrained_seed.spawn(seed))
        futures.append(run_dnabert2_untrained_seed.spawn(seed))
    for f in futures:
        f.get()
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Multi-seed sweep complete")


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT VERIFICATION (confirm untrained models have genuinely random weights)
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(image=nt_image, timeout=600)
def verify_ntv2_weights():
    """Load NTv2 pretrained and from_config, confirm weights differ."""
    import sys
    import torch
    sys.path.insert(0, "/root/project")
    from lib.adapters.nt import NTAdapter

    a_pre = NTAdapter()
    a_pre.load(pretrained=True)

    a_rand = NTAdapter()
    a_rand.load(pretrained=False)

    pre_params = dict(a_pre.model.named_parameters())
    rand_params = dict(a_rand.model.named_parameters())

    n_match = 0
    n_differ = 0
    n_shape_mismatch = 0
    n_missing = 0
    for name in sorted(pre_params.keys()):
        if name not in rand_params:
            n_missing += 1
            continue
        if pre_params[name].shape != rand_params[name].shape:
            n_shape_mismatch += 1
            print(f"  SHAPE MISMATCH: {name}: pretrained={list(pre_params[name].shape)} vs random={list(rand_params[name].shape)}")
            continue
        match = torch.allclose(pre_params[name], rand_params[name], atol=1e-6)
        if match:
            n_match += 1
        else:
            n_differ += 1

    print(f"NTv2 weight verification: {n_differ} DIFFER, {n_match} MATCH, {n_shape_mismatch} shape mismatch, {n_missing} missing")
    print(f"  Verdict: {'PASS — weights are genuinely random' if n_differ > 0 else 'FAIL — weights may be pretrained!'}")

    for name in sorted(pre_params.keys()):
        if name not in rand_params or pre_params[name].shape != rand_params[name].shape:
            continue
        if 'attention' in name and 'weight' in name and 'layer.0.' in name:
            p = pre_params[name].flatten()[:4].tolist()
            r = rand_params[name].flatten()[:4].tolist()
            print(f"  {name}:")
            print(f"    pretrained: {[round(x, 6) for x in p]}")
            print(f"    random:     {[round(x, 6) for x in r]}")
            break


@app.function(image=dnabert2_image, timeout=600)
def verify_dnabert2_weights():
    """Load DNABERT-2 pretrained and reinit, confirm weights differ and LayerNorm gamma=1."""
    import sys
    import torch
    sys.path.insert(0, "/root/project")
    from lib.adapters.dnabert2 import DNABERT2Adapter

    a_pre = DNABERT2Adapter()
    a_pre.load(pretrained=True)

    a_rand = DNABERT2Adapter()
    a_rand.load(pretrained=False)

    pre_params = dict(a_pre.model.named_parameters())
    rand_params = dict(a_rand.model.named_parameters())

    n_match = 0
    n_differ = 0
    ln_gamma_ok = True
    for name in sorted(pre_params.keys()):
        if name not in rand_params:
            continue
        match = torch.allclose(pre_params[name], rand_params[name], atol=1e-6)
        if match:
            n_match += 1
        else:
            n_differ += 1
        if 'LayerNorm' in name and 'weight' in name:
            is_ones = torch.allclose(rand_params[name], torch.ones_like(rand_params[name]), atol=1e-6)
            if not is_ones:
                ln_gamma_ok = False
                print(f"  WARNING: {name} is NOT ones: {rand_params[name].flatten()[:4].tolist()}")

    print(f"DNABERT-2 weight verification: {n_differ} tensors DIFFER, {n_match} tensors MATCH")
    print(f"  LayerNorm gamma=1 check: {'PASS' if ln_gamma_ok else 'FAIL'}")
    print(f"  Verdict: {'PASS — weights are genuinely random, LayerNorm intact' if n_differ > 0 and ln_gamma_ok else 'NEEDS INVESTIGATION'}")

    for name in ['encoder.layer.0.attention.self.Wqkv.weight',
                 'encoder.layer.6.attention.self.Wqkv.weight',
                 'encoder.layer.11.attention.self.Wqkv.weight']:
        if name in pre_params and name in rand_params:
            p = pre_params[name].flatten()[:4].tolist()
            r = rand_params[name].flatten()[:4].tolist()
            print(f"  {name}:")
            print(f"    pretrained: {[round(x, 6) for x in p]}")
            print(f"    random:     {[round(x, 6) for x in r]}")


# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH ALL (convenience)
# ═══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    """Launch all 7 models in parallel."""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Launching E1 druggability: 7 models x 16 targets")

    futures = [
        run_rnafm_e1.spawn(),
        run_rinalmo_e1.spawn(),
        run_ernierna_e1.spawn(),
        run_splicebert_e1.spawn(),
        run_utrlm_e1.spawn(),
        run_ntv2_e1.spawn(),
        run_dnabert2_e1.spawn(),
    ]

    for f in futures:
        f.get()

    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] All E1 experiments complete")
