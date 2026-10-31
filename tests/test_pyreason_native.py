"""Actual CPU engine; opt-in CI job. Synthetic/unreviewed, no legal validation."""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VLE_RUN_NATIVE_PYREASON") != "1", reason="separate native PyReason CI job"
)


def test_real_engine_qualification_and_cross_request_isolation():
    from src.pyreason_connector import run_worker, worker_transport
    from src.pyreason_graph import infer_graph
    from src.pyreason_pool import close_worker
    from src.pyreason_qualification import qualify_native
    from src.pyreason_runtime import installed_identity
    from src.resources import resource_path

    try:
        result = qualify_native(iterations=128, timeout_seconds=900, revision=os.environ.get("GITHUB_SHA"))
        if os.environ.get("VLE_NATIVE_REPORT_PATH"):
            Path(os.environ["VLE_NATIVE_REPORT_PATH"]).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        assert result["status"] == "passed", result
        assert result["native_activation_cases"] == 162 and result["native_graph_cases"] >= 5
        assert result["runtime"] == installed_identity()
        assert not result["decision_authorized"] and not result["legal_certification"]
        raw = json.loads(resource_path("examples", "pyreason-graph.json").read_text())
        with worker_transport("pooled"):
            one = infer_graph(raw)
            assert one["derived_facts"] == {"licensed": True}
            raw["facts"] = []
            two = infer_graph(raw)
            assert two["status"] == "computed" and two["derived_facts"] == {}
            assert all(v == [0, 1] for v in two["answers"].values()), "native state leaked from previous request"
            assert run_worker({"mode": "runtime_probe"}, 60)["annotations"] == installed_identity()
    finally:
        close_worker()


def test_native_api_fact_inference_and_readiness(tmp_path):
    from fastapi.testclient import TestClient

    from src.service.app import create_app
    from tests.test_production_workflows import post, source_review
    from tests.test_service import ingest, settings

    cfg = settings(tmp_path, pyreason_enabled=True, pyreason_warmup=True, pyreason_worker_mode="pooled")
    with TestClient(create_app(cfg)) as client:
        assert client.get("/health/ready").status_code == 200
        source = ingest(client, "The operator must not reproduce unless licensed.")
        source_review(client, source)
        raw = json.loads((Path(__file__).resolve().parents[1] / "examples/pyreason-graph.json").read_text())
        # Test data align to the extractor's actual predicate identifier.
        from src.formalization import formalize_candidates

        norm = formalize_candidates(source["candidates"])[0].selected()
        assert len(norm.exceptions) == 1
        raw["bindings"] = {norm.exceptions[0].name: "effective"}
        response = post(
            client,
            f"/v1/sources/{source['id']}/evaluate",
            {
                "reasoning_backend": "pyreason",
                "high_stakes": False,
                "pyreason_program": raw,
                "observed_actions": [norm.action],
            },
        )
        assert response.status_code == 200, response.text
        value = response.json()
        assert value["compliance"]["reasoning"]["backend"]["fact_inference"]["status"] == "computed"
        assert value["compliance"]["status"] == "compliant" and not value["decision_authorized"]


@pytest.mark.parametrize("arity", [1, 2])
@pytest.mark.parametrize("bound", [[1, 1], [0, 0], [0.7, 0.9], [0, 1]])
def test_native_observation_only_queries_preserve_exact_intervals(arity, bound):
    from src.pyreason_connector import worker_transport
    from src.pyreason_graph import infer_graph

    arguments = ["work"] * arity
    raw = {
        "schema": "vle-pyreason-graph/v1",
        "kind": "synthetic",
        "review_status": "unreviewed",
        "entities": ["work"],
        "predicates": {"observed": arity},
        "facts": [{"atom": {"predicate": "observed", "arguments": arguments}, "interval": bound, "start": 0, "end": 0}],
        "rules": [],
        "horizon": 0,
        "queries": [{"id": "observation", "atom": {"predicate": "observed", "arguments": arguments}, "at": 0}],
        "bindings": {"observation": "observation"},
    }
    with worker_transport("pooled"):
        result = infer_graph(raw)
    assert result["status"] == "computed", result
    assert result["answers"]["observation"] == bound
    if bound not in ([0, 0], [1, 1]):
        assert result["derived_facts"] == {}
    else:
        assert result["derived_facts"] == {"observation": bound == [1, 1]}
