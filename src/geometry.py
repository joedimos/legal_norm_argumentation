"""Bounded behavioral pseudometrics; topology is an opt-in diagnostic only."""

from __future__ import annotations

import math
import random
from itertools import product

from .audit_bundle import canonical_json
from .family_contract import parse_family
from .norm_codec import decode_norm
from .phase4 import BoundHit, Budget, envelope, fields, integer
from .redundancy.align import align
from .temporal import Formula, evaluate_formula, validate_formula
from .temporal_deontic import TemporalNorm


def geometry(request, *, horizon=None, samples=None, seed=None, topology=None):
    if request.get("schema") == "vle-family/v1":
        request = {"schema": "vle-geometry/v1", "family": request, "horizon": request["horizon"]}
    fields(
        request,
        "vle-geometry/v1",
        {
            "family",
            "interpretations",
            "horizon",
            "assumptions",
            "alignment",
            "samples",
            "seed",
            "topology",
            "max_states",
            "timeout_ms",
        },
        {"horizon"},
    )
    if ("family" in request) == ("interpretations" in request):
        raise ValueError("one family or interpretation array required")
    h = integer(horizon if horizon is not None else request["horizon"], 1, 32, "horizon")
    budget = Budget(max_states=request.get("max_states", 65536), timeout_ms=request.get("timeout_ms", 5000))
    sample = samples if samples is not None else request.get("samples")
    seed = integer(seed if seed is not None else request.get("seed", 23), 0, 2**32 - 1, "seed")
    if sample is not None:
        integer(sample, 1, 65536, "sample size")
    topology = request.get("topology", False) if topology is None else topology
    if type(topology) is not bool:
        raise ValueError("topology opt-in must be Boolean")

    def done(status, reason, **extra):
        return envelope(
            "vle-geometry-report/v1",
            request,
            status=status,
            reason=reason,
            horizon=h,
            mode="sampled" if sample is not None else "exact",
            seed=seed,
            bounds=budget.to_dict(),
            metric_scope="behavioral_equivalence_quotient_not_syntactic_identity",
            **extra,
        )

    try:
        assumptions = [Formula.from_dict(a) for a in request.get("assumptions", [])]
        if len(assumptions) > 64:
            raise ValueError("at most64 shared assumptions")
        findings = []
        forms = []
        ids = []
        names = set()
        if "family" in request:
            parsed = parse_family(request["family"])
            findings = parsed["findings"]
            if request["family"]["horizon"] != h:
                raise ValueError("family horizon cannot be silently changed")
            member_assumptions = [
                canonical_json([a.to_dict() for a in i["assumptions"]]) for i in parsed["interpretations"]
            ]
            if len(set(member_assumptions)) > 1:
                raise BoundHit("nonshared_assumptions_require_explicit_universe")
            assumptions += parsed["interpretations"][0]["assumptions"]
            forms = [i["property"] for i in parsed["interpretations"]]
            ids = [i["id"] for i in parsed["interpretations"]]
            if "behavior" in request["family"] or request["family"].get("observations"):
                raise BoundHit("response_or_observation_constraints_require_explicit_shared_formulas")
        else:
            raw = request["interpretations"]
            if not isinstance(raw, list) or not 1 <= len(raw) <= 16:
                raise ValueError("requires1-16 interpretations")
            if all("norm" in item for item in raw):
                norms, alignment = align([decode_norm(i["norm"]) for i in raw], explicit=request.get("alignment"))
                if alignment["unresolved"]:
                    raise BoundHit("unresolved_predicate_alignment")
                for n in norms:
                    if n.assumptions or n.references:
                        raise BoundHit("unmodeled_interpretation_metadata")
                    t = TemporalNorm(n, 0, h - 1)
                    forms.append(t.constraint())
                    ids.append(n.id)
                    names |= validate_formula(t.guard()) | {t.action_atom}
            else:
                if any(set(i) != {"id", "property"} for i in raw):
                    raise ValueError("explicit shared-atom formula records require id/property")
                forms = [Formula.from_dict(i["property"]) for i in raw]
                ids = [i["id"] for i in raw]
        if len(forms) > 16 or len(set(ids)) != len(ids) or any(not isinstance(id, str) or not id for id in ids):
            raise ValueError("unique1-16 ids required")
        for f in forms + assumptions:
            names |= validate_formula(f)
        names = sorted(names)
        if len(names) > 64:
            raise BoundHit("atom_limit")
        bits = len(names) * h
        if sample is None and (bits > 20 or 2**bits > budget.max_states):
            raise BoundHit("state_limit")
        # Reproducible research sampling; never credentials or authentication material.
        rng = random.Random(seed)  # nosec B311
        valuations = (
            product((False, True), repeat=bits)
            if sample is None
            else (tuple(bool(rng.getrandbits(1)) for _ in range(bits)) for _ in range(sample))
        )
        signatures = [[] for _ in forms]
        accepted = 0
        for values in valuations:
            budget.tick()
            trace = [{n: values[t * len(names) + i] for i, n in enumerate(names)} for t in range(h)]
            if not all(evaluate_formula(a, trace) is True for a in assumptions):
                continue
            accepted += 1
            for i, f in enumerate(forms):
                budget.tick()
                value = evaluate_formula(f, trace)
                if value is None:
                    raise BoundHit("unknown_complete_valuation")
                signatures[i].append(value)
        if not accepted:
            raise BoundHit("no_admissible_world_or_sample")
        counts = [
            [sum(a != b for a, b in zip(signatures[i], signatures[j], strict=True)) for j in range(len(forms))]
            for i in range(len(forms))
        ]
        classes = []
        for i, id in enumerate(ids):
            group = next((g for g in classes if counts[i][ids.index(g[0])] == 0), None)
            if group is None:
                classes.append([id])
            else:
                group.append(id)
        pairs = max(1, len(forms) * (len(forms) - 1) // 2)
        epsilon = math.sqrt(math.log(2 * pairs / 0.05) / (2 * accepted)) if sample is not None else 0
        matrix = [[v / accepted for v in row] for row in counts]
        intervals = [[[max(0, d - epsilon), min(1, d + epsilon)] for d in row] for row in matrix]
        violations = [
            (i, j, k)
            for i in range(len(forms))
            for j in range(len(forms))
            for k in range(len(forms))
            if counts[i][k] > counts[i][j] + counts[j][k]
        ]
        persistent = None
        if topology:
            from .persistence import rips

            persistent = rips(counts, accepted, budget)
        budget.tick()
        return done(
            "reported",
            "complete_in_declared_measure",
            ids=ids,
            atoms=names,
            denominator=accepted,
            disagreement_counts=counts,
            distances=matrix,
            intervals=intervals,
            confidence=0.95 if sample is not None else None,
            interval_method="simultaneous_Hoeffding_union_bound" if sample is not None else "exact",
            equivalence_classes=classes,
            equivalence_basis="sample_empirical_only" if sample is not None else "exact_bounded_behavior",
            triangle_violations=violations,
            persistence=persistent,
            topology_status="computed" if topology else "not_requested",
            review_findings=findings,
            sampling_measure="uniform_boolean_states_not_legal_likelihood",
            selected_id=None,
        )
    except (BoundHit, ValueError) as exc:
        # Unsupported legal/temporal translation is not a successful distance report.
        return done("unknown", str(exc), distances=None, intervals=None, persistence=None, selected_id=None)
