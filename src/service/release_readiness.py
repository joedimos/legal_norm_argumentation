"""Read-only deployment prerequisites. Never authorizes a case or a release."""

from datetime import UTC, datetime

from sqlalchemy import select

from src.courtlistener import IP_QUERIES
from src.resources import resource_path

from .automatic_imports import aware, configuration_hash, jobs
from .db_models import ImportCursorRecord
from .pyreason_release import pyreason_release_gate
from .release_policy import deployment_release_gate


def release_readiness(settings, database, *, now=None):
    now = now or datetime.now(UTC)
    checks = []

    def check(name, passed, required_action):
        checks.append({"check": name, "passed": bool(passed), "required_action": required_action})

    check("strict_ip_profile", settings.release_profile == "ip-supervised-v1" and settings.release_enforcement == "strict",
          "Select the ip-supervised-v1 profile and strict enforcement; legacy behavior remains available separately.")
    check("model_binding", bool(settings.build_revision and settings.model_name and settings.model_revision and settings.inference_enabled),
          "Bind the exact deployed build/model revision and enable the approved inference worker.")
    check("provider_configuration", settings.auto_import_enabled and settings.legifrance_enabled and settings.courtlistener_enabled
          and settings.has_piste_credentials and settings.courtlistener_api_token
          and bool(settings.courtlistener_api_token.get_secret_value().strip()) and settings.legifrance_piste_environment == "production",
          "Configure production PISTE and CourtListener credentials and enable both automatic IP providers.")
    try:
        for group, name in (("benchmarks", "research-v1.json"), ("schemas", "norm-interpretation.schema.json"), ("migrations", "env.py")):
            resource_path(group, name)
        assets = True
    except (OSError, ValueError):
        assets = False
    check("distribution_resources", assets, "Install the complete tested wheel or source distribution.")
    try:
        db_ready = database.ready(require_migrations=True)
    except Exception:
        db_ready = False
    check("migrated_database", db_ready, "Run forward migrations and verify database readiness.")
    required = {"legifrance:ip"} | {f"courtlistener:{t}:recent" for t in IP_QUERIES}
    active = jobs(settings)
    freshness = []
    if db_ready:
        try:
            with database.session() as session:
                records = {r.key: r for r in session.scalars(select(ImportCursorRecord).where(ImportCursorRecord.key.in_(required))).all()}
                for key in sorted(required):
                    record = records.get(key)
                    stamp = aware(record.last_success_at) if record else None
                    age = (now - stamp).total_seconds() if stamp else None
                    passed = bool(record and key in active and record.config_hash == configuration_hash(settings, active[key])
                                  and record.failure_count == 0 and age is not None
                                  and 0 <= age <= settings.release_source_max_age_seconds)
                    freshness.append({"job": key, "passed": passed, "last_success_at": stamp.isoformat() if stamp else None})
        except Exception:
            freshness = []
    check("recent_import_batches", len(freshness) == len(required) and all(x["passed"] for x in freshness),
          "Complete fresh successful batches for the French IP code and each U.S. IP topic; this does not establish corpus completeness.")
    release = deployment_release_gate(settings)
    if settings.pyreason_enabled:
        check("pyreason_backend_qualification", pyreason_release_gate(settings)["passed"],
              "Qualify this native runtime/build, persisted native load, and independently signed held-out IP benchmark.")
    check("pinned_external_evidence", release["passed"],
          "Supply current signed held-out legal annotations, model evidence, source probes and staging smoke/load/restore reports.")
    return {"schema": "vle-release-readiness/v1", "status": "prerequisites_met" if all(c["passed"] for c in checks) else "blocked",
            "version": settings.service_version, "build_revision": settings.build_revision,
            "profile": settings.release_profile, "checks": checks,
            "missing": [c["check"] for c in checks if not c["passed"]], "import_freshness": freshness,
            "observed_at": now.isoformat(), "bounds": {"max_import_jobs": 5, "provider_model_calls": 0,
                                                        "source_max_age_seconds": settings.release_source_max_age_seconds},
            "scope": "deployment_prerequisites_only_case_review_still_required",
            "decision_authorized": False, "legal_certification": False}


def main():
    import argparse
    import json

    from .config import get_settings
    from .database import Database

    argparse.ArgumentParser(description=__doc__).parse_args()
    database = None
    try:
        settings = get_settings()
        database = Database(settings)
        result = release_readiness(settings, database)
    except Exception:
        # Configuration errors can echo secrets in Pydantic/driver exceptions.
        result = {"schema": "vle-release-readiness/v1", "status": "blocked", "missing": ["valid_configuration"],
                  "reason": "configuration_or_readiness_failed", "decision_authorized": False, "legal_certification": False}
    finally:
        if database is not None:
            database.dispose()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "prerequisites_met" else 1


if __name__ == "__main__":
    raise SystemExit(main())
