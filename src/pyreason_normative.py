"""Active, required-when-selected PyReason activation; no advisory fallback."""

from __future__ import annotations

import subprocess  # nosec B404
import time
from dataclasses import replace

from .defeasible import Activation, activation
from .pyreason_connector import (
    BACKEND_VERSION,
    MAX_INPUT_BYTES,
    MAX_OUTPUT_BYTES,
    MAX_PARALLEL_WORKERS,
    ResourceBound,
    WorkerFailure,
    current_worker_mode,
    digest,
    interval,
    run_worker,
)
from .pyreason_program import LIMITS, MODE, OUTCOMES, validate_program


def compile_activations(norms, context):
    if not norms:
        raise ValueError("normative activation requires norms")
    if len(norms) > LIMITS["max_norms"]:
        raise ResourceBound("norm_count_bound")
    if len({n.id for n in norms}) != len(norms):
        raise ValueError("duplicate norms")
    nodes, rules, names = {}, [], {}
    clauses = 0
    for index, norm in enumerate(sorted(norms, key=lambda n: n.id)):
        if any(len(values) > LIMITS["max_predicates_per_list"] for values in (norm.conditions, norm.exceptions)):
            raise ResourceBound("predicate_count_bound")
        node, flags = f"n{index}", {}
        names[node] = norm.id

        def state(label, value, flags=flags):
            for suffix, expected in (("t", True), ("f", False), ("u", None)):
                flags[f"{label}_{suffix}"] = value is expected

        effective = norm.authority.is_effective(context.at_date)
        state("scope", effective if norm.authority.effective_from or norm.authority.effective_to else True)
        state("actor", context.actor is None or context.actor == norm.subject)
        for i, predicate in enumerate(norm.conditions):
            state(f"c{i}", predicate.evaluate(context.facts))
        for i, predicate in enumerate(norm.exceptions):
            state(f"e{i}", predicate.evaluate(context.facts))
        nodes[node] = flags

        def rule(head, body, node=node):
            nonlocal clauses
            clauses += len(body) + 1
            if len(rules) >= LIMITS["max_rules"] or clauses > LIMITS["max_clauses"]:
                raise ResourceBound("program_size_bound")
            rules.append({"node": node, "head": head, "body": [[label, value] for label, value in body]})

        # These guards preserve the reference evaluator's ordered short-circuit semantics.
        rule("inactive_temporal", [("scope_f", True)])
        rule("unknown_date", [("scope_u", True)])
        rule("inactive_subject", [("scope_t", True), ("actor_f", True)])
        base = [("scope_t", True), ("actor_t", True)]
        true_conditions = [(f"c{i}_t", True) for i in range(len(norm.conditions))]
        no_false_conditions = [(f"c{i}_f", False) for i in range(len(norm.conditions))]
        for i in range(len(norm.conditions)):
            rule("inactive_condition", [*base, (f"c{i}_f", True)])
            rule("unknown_condition", [*base, *no_false_conditions, (f"c{i}_u", True)])
        no_true_exceptions = [(f"e{i}_t", False) for i in range(len(norm.exceptions))]
        for i in range(len(norm.exceptions)):
            rule("exception_applies", [*base, *true_conditions, (f"e{i}_t", True)])
            rule("unknown_exception", [*base, *true_conditions, *no_true_exceptions, (f"e{i}_u", True)])
        rule("active", [*base, *true_conditions, *[(f"e{i}_f", True) for i in range(len(norm.exceptions))]])
    program = {"mode": MODE, "nodes": nodes, "rules": rules, "timesteps": 1}
    validate_program(program)
    return program, names


def unknown_report(reason, timeout_seconds=60):
    return {
        "schema": "vle-pyreason-normative/v1",
        "status": "unknown",
        "role": "required_activation_backend",
        "semantics": MODE,
        "decision_authorized": False,
        "legal_certification": False,
        "bounds": {
            **LIMITS,
            "timesteps": 1,
            "timeout_seconds": timeout_seconds
            if type(timeout_seconds) is int and 1 <= timeout_seconds <= 300
            else None,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "max_input_bytes": MAX_INPUT_BYTES,
            "max_parallel_workers": MAX_PARALLEL_WORKERS,
        },
        "activations": {},
        "reason": reason,
    }


def run_activation_backend(norms, context, *, timeout_seconds=60, fact_program=None):
    report = unknown_report("computation_not_completed", timeout_seconds)
    started = time.monotonic()
    try:
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 300:
            raise ValueError("invalid timeout")
        if fact_program is not None:
            from .pyreason_graph import infer_graph
            inference = infer_graph(fact_program, timeout_seconds=timeout_seconds)
            report["fact_inference"] = inference
            if inference["status"] != "computed":
                report["reason"] = "required_fact_inference_incomplete"
                return report
            if any(k in context.facts and context.facts[k] != v for k, v in inference["derived_facts"].items()):
                report["reason"] = "derived_observation_conflict"
                return report
            context = replace(context, facts={**context.facts, **inference["derived_facts"]})
        program, names = compile_activations(norms, context)
        report.update(
            input_sha256=digest(program),
            argument_map_sha256=digest(names),
            norm_semantic_hashes={n.id: n.semantic_hash() for n in sorted(norms, key=lambda n: n.id)},
            context_sha256=digest(
                {
                    "facts": dict(context.facts),
                    "actor": context.actor,
                    "at_date": context.at_date,
                    "observed_actions": list(context.observed_actions),
                }
            ),
            program=program,
            node_map=names,
        )
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise ResourceBound("backend_total_time_bound")
        output = run_worker(program, remaining if fact_program is not None else timeout_seconds)
        annotations = output["annotations"]
        if not isinstance(annotations, dict) or set(annotations) != set(OUTCOMES):
            raise ValueError("missing activation outcomes")
        for values in annotations.values():
            if not isinstance(values, dict) or set(values) != set(names):
                raise ValueError("incomplete activation outcomes")
            if any(interval(value) not in ([0, 1], [1, 1]) for value in values.values()):
                raise ValueError("unexpected outcome interval")
        derived = {}
        for node, id in names.items():
            outcomes = [key for key, values in annotations.items() if interval(values[node]) == [1, 1]]
            if len(outcomes) != 1:
                raise ValueError("missing or inconsistent activation")
            status, reason = OUTCOMES[outcomes[0]]
            derived[id] = Activation(id, status, (reason,))
        # Independent check is a blocking integrity gate, not a silent substitute.
        disagreements = [n.id for n in norms if derived[n.id] != activation(n, context)]
        if disagreements:
            report.update(
                reason="backend_disagreement",
                disagreement_norm_ids=sorted(disagreements),
                disagreements=[
                    {
                        "norm_id": n.id,
                        "pyreason_status": derived[n.id].status,
                        "reference_status": activation(n, context).status,
                    }
                    for n in sorted(norms, key=lambda n: n.id)
                    if n.id in disagreements
                ],
            )
            return report
        if time.monotonic() - started >= timeout_seconds:
            raise ResourceBound("backend_total_time_bound")
        report.update(
            status="computed",
            backend_version=BACKEND_VERSION,
            worker_mode=current_worker_mode(),
            independent_check="matched",
            activations={
                key: {"norm_id": key, "status": value.status, "reasons": list(value.reasons)}
                for key, value in sorted(derived.items())
            },
            annotations=annotations,
        )
        report.pop("reason", None)
    except subprocess.TimeoutExpired:
        report.update(reason="backend_timeout", bound_hit=True)
    except ResourceBound as exc:
        report.update(reason=str(exc), bound_hit=True)
    except WorkerFailure as exc:
        report["reason"] = str(exc)
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OSError,
        OverflowError,
        RecursionError,
        subprocess.SubprocessError,
    ):
        report["reason"] = "invalid_or_incomplete_backend_computation"
    return report
