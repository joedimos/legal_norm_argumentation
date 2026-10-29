"""Conservative bounded guard for the documented /v1/ API surface.

This is a compatibility floor, not a proof of behavioral equivalence. Existing
descriptions, defaults, constraints and schema references are frozen. Optional
properties/parameters, new paths and unused components may be added. Other
changes require review; equivalent schema rewrites can therefore be flagged.
"""

from copy import deepcopy

from .audit_bundle import MAX_BYTES, canonical_json, digest
from .model_runtime import strict_json
from .resources import resource_path

BASELINE_SHA256 = "e7ab2ebd6cab3d069ac6a83cdc0c8d1672a7d7f30c376b368abcbbf0c160545a"


def current_contract():
    from .service.config import Settings
    from .service.factory import create_app

    app = create_app(Settings(environment="test", database_url="sqlite+pysqlite:///:memory:",
                              auto_create_schema=False, api_keys_json="[]", _env_file=None))
    try:
        value = app.openapi()
        value["paths"] = {p: v for p, v in value["paths"].items() if p.startswith("/v1/")}
        value.pop("info", None)  # Build version/marketing text are not request contracts.
        candidates, traversed = list(app.routes), 0
        while candidates:
            route = candidates.pop()
            traversed += 1
            if traversed > 1000:
                raise ValueError("route_inventory_bound")
            if hasattr(route, "effective_candidates"):
                # FastAPI 0.143 retains included routers as effective contexts.
                candidates.extend(route.effective_candidates())
                continue
            if getattr(route, "path", None) not in value["paths"] or getattr(route, "dependant", None) is None:
                continue
            roles = []
            pending, seen = list(route.dependant.dependencies), set()
            while pending:
                dependency = pending.pop()
                if id(dependency) in seen:
                    continue
                seen.add(id(dependency))
                if len(seen) > 1000:
                    raise ValueError("dependency_inventory_bound")
                declared = getattr(dependency.call, "vle_required_roles", None)
                if declared is not None:
                    roles.append(list(declared))
                pending.extend(dependency.dependencies)
            for method in route.methods:
                operation = value["paths"][route.path].get(method.lower())
                if operation is not None:
                    operation["x-vle-role-guards"] = sorted(roles)
        if any("x-vle-role-guards" not in op for path in value["paths"].values() for op in path.values()):
            raise ValueError("uninspected_API_operation")
        return value
    finally:
        app.state.db.dispose()


def compare_contracts(baseline, current, *, max_nodes=50000, max_changes=100, max_depth=64):
    bounds = {"max_nodes": max_nodes, "max_changes": max_changes, "max_depth": max_depth}
    result = {"schema": "vle-api-contract-report/v1", "status": "unknown", "changes": [], "bounds": bounds,
              "scope": "documented_shapes_and_declared_role_guards_not_behavioral_equivalence",
              "decision_authorized": False, "legal_certification": False}
    count = 0
    changes = result["changes"]

    def tick(depth):
        nonlocal count
        count += 1
        if count > max_nodes or depth > max_depth:
            raise ValueError("contract_analysis_bound")

    def validate(value, document, depth=0):
        tick(depth)
        if isinstance(value, dict):
            if "$ref" in value:
                ref = value["$ref"]
                if not isinstance(ref, str) or not ref.startswith("#/components/"):
                    raise ValueError("unsupported_external_contract_reference")
                target = document
                for part in ref[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
            for child in value.values():
                validate(child, document, depth + 1)
        elif isinstance(value, list):
            for child in value:
                validate(child, document, depth + 1)

    def changed(path):
        if len(changes) >= max_changes:
            raise ValueError("contract_change_bound")
        changes.append({"path": path, "reason": "existing_contract_changed_or_new_required_constraint"})

    def walk(old, new, path="", depth=0):
        tick(depth)
        if type(old) is not type(new):
            changed(path)
        elif isinstance(old, dict):
            if (old.get("additionalProperties") is False and isinstance(old.get("properties"), dict)
                    and isinstance(new.get("properties"), dict) and new["properties"].keys() - old["properties"].keys()):
                changed(path + "/properties")
            for key in sorted(old):
                if key not in new:
                    changed(path + "/" + key)
                else:
                    walk(old[key], new[key], path + "/" + key, depth + 1)
            for key in sorted(new.keys() - old.keys()):
                # Only these extension points are automatically accepted.
                if path in {"/paths", "/components/schemas"}:
                    continue
                if path.endswith("/properties"):
                    continue  # A changed required list is checked separately.
                if key == "parameters" and isinstance(new[key], list) and all(p.get("required", False) is False for p in new[key]):
                    continue
                if path.startswith("/paths/") and key in {"get", "post", "put", "patch", "delete", "head", "options"}:
                    continue
                changed(path + "/" + key)
        elif isinstance(old, list):
            if path.endswith("/parameters"):
                old_by = {(p["in"], p["name"]): p for p in old}
                new_by = {(p["in"], p["name"]): p for p in new}
                if len(old_by) != len(old) or len(new_by) != len(new):
                    raise ValueError("duplicate_contract_parameter")
                for key, value in old_by.items():
                    if key not in new_by:
                        changed(path + "/" + key[1])
                    else:
                        walk(value, new_by[key], path + "/" + key[1], depth + 1)
                for key in sorted(new_by.keys() - old_by.keys()):
                    if new_by[key].get("required", False) is not False:
                        changed(path + "/" + key[1])
            elif old != new:
                changed(path)
        elif old != new:
            changed(path)

    try:
        if (any(type(v) is not int or v < 1 for v in bounds.values()) or max_nodes > 200000 or max_depth > 128
                or not isinstance(baseline, dict) or not isinstance(current, dict)
                or not baseline.get("paths") or not current.get("paths")
                or any(not str(d.get("openapi", "")).startswith("3.1.") for d in (baseline, current))):
            raise ValueError("invalid_contract_or_bounds")
        methods = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
        for document in (baseline, current):
            for path, operations in document["paths"].items():
                if not path.startswith("/v1/") or not isinstance(operations, dict) or not operations or set(operations) - methods:
                    raise ValueError("unsupported_path_item")
                for operation in operations.values():
                    if not isinstance(operation, dict) or not isinstance(operation.get("responses"), dict):
                        raise ValueError("unsupported_operation")
        validate(baseline, baseline)
        validate(current, current)
        walk(deepcopy(baseline), deepcopy(current))
        result.update(status="review_required" if changes else "compatible_with_baseline", truncated=False,
                      baseline_sha256=digest(canonical_json(baseline)), current_sha256=digest(canonical_json(current)))
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        result.update(status="unknown", reason="invalid_unsupported_or_bounded_contract", truncated=True)
    result["nodes_checked"] = count
    return result


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", help="Compare a recorded OpenAPI JSON instead of inspecting the installed app")
    args = parser.parse_args()
    baseline = strict_json(resource_path("schemas", "api-v1-openapi.json").read_bytes())
    if digest(canonical_json(baseline)) != BASELINE_SHA256:
        parser.error("frozen baseline hash changed; use an explicitly reviewed contract version")
    try:
        if args.current:
            with open(args.current, "rb") as source:
                raw = source.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("contract input bound")
            current = strict_json(raw)
        else:
            current = current_contract()
        report = compare_contracts(baseline, current)
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        report = {"schema": "vle-api-contract-report/v1", "status": "unknown", "reason": "contract_unavailable_or_invalid",
                  "decision_authorized": False, "legal_certification": False}
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "compatible_with_baseline" else 1


if __name__ == "__main__":
    raise SystemExit(main())
