from pathlib import Path

import pytest

from src.research_benchmark import run_benchmarks


@pytest.mark.parametrize("backend", ["exhaustive", "z3"])
def test_versioned_corpus_all_families(backend):
    if backend == "z3":
        pytest.importorskip("z3")
    report = run_benchmarks(Path(__file__).parents[1] / "benchmarks/research-v1.json", backend=backend)
    assert report["all_passed"], [row for row in report["cases"] if not row["passed"]]
    assert report["case_count"] == 62
    assert len(report["by_family"]) == 6
    assert len(report["corpus_sha256"]) == 64
