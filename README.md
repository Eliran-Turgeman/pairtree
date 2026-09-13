<h1 align="center">PairTree</h1>

<p align="center">
  A standalone research continuation investigating predecessor-conditioned
  allocation for diffusion draft trees.
</p>

<p align="center">
  Built on the official <strong>DDTree (Diffusion Draft Tree)</strong>
  implementation by Liran Ringel and Yaniv Romano.
</p>

<p align="center">
  <a href="https://liranringel.github.io/ddtree/">🌐 Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2604.12989">📄 Paper</a>
</p>

## What this repository contains

This repository preserves the implementation and publishable analysis for
using DFlash2 predecessor-conditioned selector scores to allocate a
multi-branch DDTree. The complete raw empirical record remains available in
the maintainers' local evidence archive. This is intentionally a new
standalone repository rather than a fork with inherited Git history.

The main frozen result is:

- within the same DFlash2 system, Pairwise-K16 improves end-to-end throughput
  over released unary DDTree by 5.3%-9.6% across GSM8K, MATH500, HumanEval,
  and controlled/native MT-Bench on one H100;
- the tested DFlash2+Pairwise system remains slower in absolute tokens/s than
  released original DFlash and original DFlash+DDTree.

Start with:

- [`research_notes/pairwise_ddtree_formalization.md`](research_notes/pairwise_ddtree_formalization.md)
  for the algorithm and mathematical claims;
- [`research_notes/dflash2_step9_4b_throughput_validation.md`](research_notes/dflash2_step9_4b_throughput_validation.md)
  for the final empirical interpretation;
- [`research_notes/step9_4b_benchmark_report.pdf`](research_notes/step9_4b_benchmark_report.pdf)
  for the complete illustrated benchmark report;
- [`EVIDENCE.md`](EVIDENCE.md) for the artifact map and integrity checks.

Raw benchmark artifacts under `artifacts/`, `runs/`, `traces/`, and `logs/`
are intentionally excluded from Git. They remain in the local research
archive and can be published separately later. The repository tracks the
derived analysis, reports, research notes, and integrity manifests needed to
inspect the reported conclusions without downloading roughly 1.8 GiB of raw
tensors.

## Attribution and scope

DDTree supplies the verification-tree framework, best-first construction,
target verifier, and original unary allocation. DFlash2 supplies the drafter,
candidate lattice, and predecessor-conditioned selector machinery. This
research continuation contributes the mass-preserving conditional allocation
formulation, its integration with the DDTree builder/verifier, and the
controlled falsification and validation experiments documented here.

This repository does not claim ownership of DDTree or DFlash2, and does not
claim that the tested Pairwise-DFlash2 system is faster than released
original-DFlash systems.

## Setup

This codebase is intended for a CUDA-enabled PyTorch environment.

```bash
pip install -r requirements.txt
```

## Run Experiments

```bash
bash run_benchmark.sh
```

This produces benchmark outputs in `runs/` and logs in `logs/`.

To run a limited GSM8K benchmark on one Lambda Cloud GPU:

```bash
EXPERIMENT="2026-09-02_initial-reproduction_a100-40gb_gsm8k-32"

bash run_benchmark.sh \
  --gpus 0 \
  --task gsm8k:32 \
  --model-draft-pair 'Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16' \
  --temperature 0.0 \
  --mode sdpa \
  --run-dir "runs/${EXPERIMENT}" \
  --log-dir "logs/${EXPERIMENT}"
```

This runs the baseline, DFlash, and DDTree methods for 32 examples using one
worker. Give each experiment the same descriptive subfolder name under
`runs/` and `logs/` so its artifacts remain paired. Use
`bash run_benchmark.sh --help` to see all sweep parameters. The model weights,
dataset, and FlashAttention dependency are downloaded on the first run, so the
Lambda instance needs Hugging Face access and enough local storage for both
models.

Summarize a single benchmark artifact and export sample-level data to CSV:

```bash
python3 summarize_run.py runs/<experiment-name>/<run-name>.pt
```

The CSV is written next to the `.pt` file by default. Use `--csv <path>` to
choose a different output location.

## Unary preparation correction

New direct `benchmark.py` invocations default to
`--dflash2-unary-preparation lean`: unary trees skip the conditional selector
and pairwise lattice they do not use. The checkpoint, unary scores, candidate
support, allocation rule, and verifier are unchanged. Pairwise still computes
its conditional scores. `shared` reproduces the old preparation path; `both`
adds explicitly named `_shared_tb*` unary controls to the same timed process.

The frozen Step-9 report measured shared preparation, so its 5.3%-9.6%
throughput gain is not yet a result against lean unary. Its existing results
and artifacts have not been replaced. The Step-8 and existing Step-9 launcher
profiles explicitly retain shared preparation for reproduction.

The focused correction run needs one H100 80GB with the original CUDA
environment and a clean, committed checkout:

```bash
bash run_step9_4b_throughput.sh unary-smoke
bash run_step9_4b_throughput.sh unary-audit
```

Run the smoke first. It checks both families and the lean/shared controls on
two prompts. The audit uses 32 GSM8K and 32 HumanEval prompts, B=16/64,
256 output tokens, three timing repetitions, and two independent launches
with opposite family order. Both commands generate paired-bootstrap analysis
under their own `artifacts/step9_4b/<commit>/<profile>/analysis/` directory.
Lean/shared output or per-round acceptance disagreement aborts the audit.
Timing instrumentation is deliberately unchanged to isolate the preparation
correction; this is not an uninstrumented serving benchmark.

See `research_notes/step9_4b_throughput_protocol.md` for the audit endpoints.

## DFlash2 Proof of Concept

Verify that the experimental
[`mgoin/Qwen3-4B-speculator.dflash2`](https://huggingface.co/mgoin/Qwen3-4B-speculator.dflash2)
checkpoint generates the same greedy tokens as `Qwen/Qwen3-4B`:

```bash
python3 run_dflash2_smoke.py
```

The checkpoint uses seven speculative tokens, unary top-16 candidates, and a
rank-256 predecessor-conditioned selector. It was trained with an experimental
Speculators-native objective and is not an official reproduction of Inco's
unpublished DFlash2 training recipe.

`DFlash2Proposal` preserves the selector inputs without normalization:

- `candidate_ids`: top-16 token IDs at each speculative position
- `unary_scores`: raw unary logits for those candidates
- `anchor_pairwise_corrections`: raw anchor-to-first-position corrections
- `pairwise_corrections`: raw corrections for every adjacent top-16 candidate
  pair, with shape `[batch, positions - 1, 16, 16]`
- `corrected_scores`: unary plus pairwise scores along the selected greedy path

Run the 32-sample GSM8K DFlash2 benchmark:

```bash
EXPERIMENT="2026-09-03_dflash2_a100-40gb_gsm8k-32"

bash run_benchmark.sh \
  --gpus 0 \
  --task gsm8k:32 \
  --model-draft-pair 'Qwen/Qwen3-4B|mgoin/Qwen3-4B-speculator.dflash2' \
  --draft-type dflash2 \
  --temperature 0.0 \
  --mode sdpa \
  --run-dir "runs/${EXPERIMENT}" \
  --log-dir "logs/${EXPERIMENT}"
```

Each sample is generated by both the sequential Qwen3-4B baseline and DFlash2.
Exact token matches are recorded in the artifact and exported by
`summarize_run.py`. DFlash2 is lossless at the decoding-algorithm level, but
BF16 block verification can occasionally choose a different argmax near a
logit tie because it uses different matrix shapes than one-token decoding.
The summary reports round-weighted tokens per decoding round, including the
verifier-carried token. Subtract one to obtain matched speculative tokens.

### Collect DFlash2 research traces

Collect the reproducible 32-prompt GSM8K trace used for offline unary-versus-
pairwise tree-selection experiments:

```bash
python3 collect_dflash2_traces.py \
  traces/2026-09-03_dflash2_a100-40gb_gsm8k-32/gsm8k__Qwen_Qwen3-4B__mgoin_Qwen3-4B-speculator.dflash2__seed0__traces.pt \
  --max-samples 32 \
  --seed 0
```

Inspect and validate the saved candidate lattice, selector scores, target
observations, and prompt metadata:

```bash
python3 inspect_dflash2_traces.py \
  traces/2026-09-03_dflash2_a100-40gb_gsm8k-32/gsm8k__Qwen_Qwen3-4B__mgoin_Qwen3-4B-speculator.dflash2__seed0__traces.pt
```

The collector does not make extra target-model calls and does not construct a
DDTree. See `research_notes/dflash2_trace_schema.md` for the exact tensor
definitions and target-token validity rules.

### Evaluate unary and pairwise trees offline

Run the frozen-trace UnaryTree versus PairwiseTree evaluation on DFlash2's
saved top-16 candidate lattice:

```bash
python3 analyze_dflash2_trees.py \
  traces/2026-09-03_dflash2_a100-40gb_gsm8k-32/gsm8k__Qwen_Qwen3-4B__mgoin_Qwen3-4B-speculator.dflash2__seed0__traces.pt \
  analysis/2026-09-03_dflash2_unary-vs-pairwise \
  --bootstrap-samples 10000
```

This analysis performs no model inference and does not modify the online
decoder. `Unary-FullMass` uses the full-vocabulary unary normalizer but remains
restricted to the saved top-16 token IDs, so it is not unrestricted DDTree at
budgets above 16. See
`research_notes/dflash2_offline_tree_evaluation.md` for the scorer definitions,
results, and go/no-go recommendation.

### Run the controlled online DFlash2 tree experiment

The Step-5 prototype compares DFlash2 greedy decoding with Unary-K16 and
Pairwise-K16 trees while reusing DDTree's existing target verifier:

```bash
bash run_benchmark.sh \
  --gpus 0 \
  --task gsm8k:32 \
  --model-draft-pair \
    'Qwen/Qwen3-4B|mgoin/Qwen3-4B-speculator.dflash2' \
  --draft-type dflash2 \
  --tree-budget 7,8,16,32,64 \
  --temperature 0.0 \
  --mode sdpa \
  --max-new-tokens 2048 \
  --run-dir runs/2026-09-03_step5-online_a100_gsm8k-32 \
  --log-dir logs/2026-09-03_step5-online_a100_gsm8k-32
```

Export prompt- and round-level measurements:

```bash
python summarize_run.py \
  runs/2026-09-03_step5-online_a100_gsm8k-32/*.pt
```

Generate prompt-bootstrap acceptance, throughput, timing, and
offline-versus-online comparisons:

```bash
python analyze_online_dflash2.py \
  runs/2026-09-03_step5-online_a100_gsm8k-32/*.pt \
  analysis/2026-09-03_step5-online_a100_gsm8k-32 \
  --bootstrap-samples 10000
```

The benchmark records `matched_draft_tokens`,
`committed_tokens_this_round`, and `verifier_bonus_committed` separately for
every online round. CUDA synchronization is used at each timing boundary.

## Run the final one-H100 throughput protocol

The frozen Step 9 protocol compares both drafter families under one process,
one visible H100, pinned model/dataset revisions, randomized method order, and
repeated per-prompt timings:

```bash
CUDA_VISIBLE_DEVICES=0 FULL_TIMING_REPETITIONS=3 \
  bash run_step9_4b_throughput.sh full
```

Run `one`, `matrix`, `domains`, and `stability` before `full` when validating a
new machine. The launcher refuses a dirty tracked worktree and writes artifacts
under `artifacts/step9_4b/<commit>/<profile>/`. The full protocol is expensive;
the limited commands above remain the quickest way to validate a setup.

Analyze a paired dataset with:

```bash
python analyze_step9_4b_throughput.py \
  --expected-commit "$(git rev-parse HEAD)" \
  --pair gsm8k=artifacts/step9_4b/<commit>/full/gsm8k_original_controlled.pt:artifacts/step9_4b/<commit>/full/gsm8k_dflash2_controlled.pt \
  --output-dir analysis/my-step9-run
```

Repeat `--pair` for MATH500, HumanEval, and MT-Bench. The analyzer labels
same-drafter comparisons as controlled and different-drafter comparisons as
cross-drafter so they cannot be interpreted as the same causal claim.

## Reproduce report artifacts

Generate the plots:

```bash
python3 plot_results.py
```

Generate the LaTeX table:

```bash
python3 make_latex_table.py
```

## Citation

```bibtex
@article{ringel2026ddtree,
  title={Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author={Ringel, Liran and Romano, Yaniv},
  journal={arXiv preprint arXiv:2604.12989},
  year={2026}
}
```
