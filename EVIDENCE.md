# Evidence map

This standalone repository was created from the working tree of
`Eliran-Turgeman/ddtree` at commit
`357b769006af4d96c6d8a4a7a806e06edaf197c3`. The new repository intentionally
does not inherit the fork's Git history. Its public snapshot preserves the
implementation, derived analysis, research record, and provenance. Raw outputs
remain in a local evidence archive and are not part of the Git history.

## Recommended reading order

1. `research_notes/pairwise_ddtree_formalization.md` — formal algorithm,
   notation, attribution, and proof boundary.
2. `research_notes/dflash2_step62_cross_domain_validation.md` — controlled
   cross-domain DFlash2 comparison.
3. `research_notes/dflash2_step63_wider_unary.md` — wider-unary falsification.
4. `research_notes/dflash2_step7_27b_validation.md` — official 27B checkpoint
   validation.
5. `research_notes/dflash2_step9_4b_throughput_validation.md` — final one-H100
   throughput interpretation.
6. `research_notes/step9_4b_benchmark_report.pdf` — illustrated full report.

## Evidence directories

| Path | Contents |
|---|---|
| `analysis/` | Compact and detailed analysis tables, provenance, confidence intervals, quality results, and manifests |
| `runs/` (local only) | Frozen pre-Step-9 benchmark tensors and exported per-prompt/per-round CSVs |
| `artifacts/step9_4b/` (local only) | Step-9 smoke, matrix, domain, stability, and full dual-drafter artifacts |
| `logs/` (local only) | Console logs for every retained empirical stage |
| `traces/` (local only) | Frozen DFlash2 candidate-lattice trace used by the offline experiments |
| `research_notes/` | Protocols, formalization, frozen decisions, reports, figures, and editable sources |

The evidence progresses from the original DDTree reproduction through DFlash2
tracing, offline allocation, online 4B validation, wider-unary falsification,
official 27B validation, and the final dual-drafter H100 throughput benchmark.

## Integrity

Important existing manifests include:

- `analysis/2026-09-05_step7-27b/artifact_manifest.sha256`
- `analysis/2026-09-05_step7-27b/raw_artifacts.sha256`
- `analysis/step9-4b-throughput/artifact_manifest.sha256`
- `analysis/step9-4b-throughput/raw_artifacts_sha256.txt`
- `analysis/step9-4b-throughput/analysis_sha256.txt`
- `analysis/step9-4b-throughput/complete_remote_sha256.txt`
- `evidence_manifest.sha256` — complete evidence snapshot in this repository

The Step-9 `complete_remote_sha256.txt` covers all 138 files copied from the
H100. Those files were verified locally before the instance was released.
`evidence_manifest.sha256` covers the complete local `analysis/`, `artifacts/`,
`logs/`, `runs/`, `traces/`, and `research_notes/` trees, excluding itself.
It is retained publicly as an inventory of the archived raw evidence.

On a machine containing the local raw archive, verify it from the repository
root:

```powershell
$failed = 0
Get-Content evidence_manifest.sha256 | ForEach-Object {
  if ($_ -match '^([0-9a-f]{64})  (.+)$') {
    $expected = $Matches[1]
    $path = $Matches[2] -replace '/', '\'
    $actual = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
      Write-Error "Hash mismatch: $path"
      $failed = 1
    }
  }
}
exit $failed
```

## Published versus local evidence

The Git repository publishes:

- source code and tests;
- `analysis/` tables, quality results, provenance, and manifests;
- research notes, figures, and final reports;
- the complete local-evidence hash manifest.

The local working copy additionally retains `artifacts/`, `runs/`, `traces/`,
and `logs/`. These paths are ignored and absent from every public commit.
Final PDFs continue to use Git LFS.

## Deliberately excluded

The migration excludes the old repository's `.git/` history, Python virtual
environments, Python/test/linter caches, editor state, LaTeX build
intermediates, and accidental empty path artifacts. Raw benchmark directories
remain on the maintainer's machine but are deliberately excluded from Git.
