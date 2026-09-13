import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def bash_executable() -> str:
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).parent.parent / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
    elif shutil.which("bash"):
        return shutil.which("bash")
    pytest.skip("Bash is required for launcher contract tests")


def run_launcher(tmp_path: Path, profile: str, *, dirty: bool = False):
    script = (ROOT / "run_step9_4b_throughput.sh").read_text(encoding="utf-8")
    stubs = """
git() {
  case "$1" in
    status) if [[ "$MOCK_DIRTY" == "yes" ]]; then echo " M benchmark.py"; fi ;;
    rev-parse) printf '%s\\n' "$MOCK_COMMIT" ;;
    *) return 1 ;;
  esac
}
nvidia-smi() { echo "GPU 0: mock H100"; }
python() { { printf '%s\\t' "$@"; printf '\\n'; } >> calls.txt; }
"""
    result = subprocess.run(
        [bash_executable(), "-s", "--", profile],
        input=stubs + script,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "OUTPUT_ROOT": "outputs",
            "MOCK_COMMIT": COMMIT,
            "MOCK_DIRTY": "yes" if dirty else "no",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )
    log = tmp_path / "calls.txt"
    calls = (
        [line.rstrip("\t").split("\t") for line in log.read_text().splitlines()]
        if log.exists() else []
    )
    return result, calls


@pytest.mark.parametrize(
    ("profile", "benchmark_count", "preparation", "samples", "repetitions"),
    [
        ("matrix", 2, "shared", "2", "1"),
        ("unary-smoke", 2, "both", "2", "1"),
        ("unary-audit", 8, "both", "32", "3"),
    ],
)
def test_launcher_matrix_and_analysis(
    tmp_path, profile, benchmark_count, preparation, samples, repetitions
) -> None:
    result, calls = run_launcher(tmp_path, profile)
    assert result.returncode == 0, result.stderr
    benchmarks = [call for call in calls if call[0] == "benchmark.py"]
    assert len(benchmarks) == benchmark_count
    for call in benchmarks:
        assert "--collect-allocation-data" not in call
        assert call[call.index("--max-samples") + 1] == samples
        assert call[call.index("--timing-repetitions") + 1] == repetitions
        assert call[call.index("--model-revision") + 1] == (
            "1cfa9a7208912126459214e8b04321603b3df60c"
        )
        if call[call.index("--draft-type") + 1] == "dflash2":
            assert call[call.index("--dflash2-unary-preparation") + 1] == preparation
            if profile.startswith("unary-"):
                assert call[call.index("--dflash2-tree-configs") + 1] == (
                    "dflash2_original_ddtree:16,64;dflash2_pairwise_k16:16,64"
                )
    if profile.startswith("unary-"):
        analysis = calls[-1]
        assert analysis[0] == "analyze_step9_4b_throughput.py"
        assert "--allow-partial" in analysis
        assert analysis.count("--pair") == benchmark_count // 2
        assert analysis[analysis.index("--bootstrap-samples") + 1] == "10000"
    if profile == "unary-audit":
        assert [call[call.index("--draft-type") + 1] for call in benchmarks] == [
            "dflash", "dflash2", "dflash2", "dflash",
            "dflash", "dflash2", "dflash2", "dflash",
        ]


def test_launcher_refuses_dirty_worktree(tmp_path) -> None:
    result, calls = run_launcher(tmp_path, "unary-smoke", dirty=True)
    assert result.returncode != 0
    assert "dirty tracked worktree" in result.stderr
    assert not calls


def test_launcher_preserves_completed_artifacts(tmp_path) -> None:
    output = tmp_path / "outputs" / COMMIT / "unary-smoke" / "gsm8k_original_controlled.pt"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing evidence")
    result, calls = run_launcher(tmp_path, "unary-smoke")
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert not calls
    assert output.read_bytes() == b"existing evidence"
