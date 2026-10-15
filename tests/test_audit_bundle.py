import io
import json
import subprocess
import sys
import zipfile

import pytest

from src.audit_bundle import canonical_json, create_bundle, verify_bundle


def test_bundle_is_deterministic_and_trusted_digest_is_distinct_from_integrity():
    artifacts = {"source.json": {"text": "Règle: α"}, "result.json": {"legal_certification": False}}
    first, digest = create_bundle(artifacts, metadata={"run_id": "run-1"})
    second, same = create_bundle(dict(reversed(list(artifacts.items()))), metadata={"run_id": "run-1"})
    assert first == second and digest == same
    report = verify_bundle(first)
    assert report["valid"] and not report["trusted_digest_matched"] and not report["signature_verified"]
    assert verify_bundle(first, expected_manifest_digest=digest)["trusted_digest_matched"]
    assert not verify_bundle(first, expected_manifest_digest="0" * 64)["valid"]


def rewritten(data, mutate):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(out, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            new_info, payload = mutate(info, payload)
            target.writestr(new_info, payload)
    return out.getvalue()


def test_modified_artifact_and_rebuilt_manifest_cannot_match_trusted_digest():
    original, trusted = create_bundle({"result.json": {"authorized": False}})
    modified = rewritten(original, lambda info, payload: (info, b'{"authorized":true}')
                         if info.filename == "result.json" else (info, payload))
    assert not verify_bundle(modified)["valid"]
    replacement, _ = create_bundle({"result.json": {"authorized": True}})
    assert verify_bundle(replacement)["valid"]  # Integrity alone cannot establish origin.
    assert not verify_bundle(replacement, expected_manifest_digest=trusted)["valid"]


@pytest.mark.parametrize("name", ["../x.json", "/x.json", "x/y.json", "manifest.json", "X.json"])
def test_unsafe_artifact_names_are_rejected(name):
    with pytest.raises(ValueError):
        create_bundle({name: {}})


def test_path_traversal_duplicates_symlinks_and_unknown_schema_rejected():
    archive, _ = create_bundle({"data.json": {"x": 1}})

    def traversal(info, data):
        if info.filename == "data.json":
            info.filename = "../data.json"
        return info, data

    assert not verify_bundle(rewritten(archive, traversal))["valid"]

    def symlink(info, data):
        info.external_attr = 0o120777 << 16
        return info, data

    assert not verify_bundle(rewritten(archive, symlink))["valid"]

    def schema(info, data):
        if info.filename == "manifest.json":
            doc = json.loads(data)
            doc["schema"] = "future/999"
            data = canonical_json(doc)
        return info, data

    assert not verify_bundle(rewritten(archive, schema))["valid"]
    out = io.BytesIO(archive)
    with zipfile.ZipFile(out, "a") as target, pytest.warns(UserWarning):
        target.writestr("data.json", b"{}")
    assert not verify_bundle(out.getvalue())["valid"]


def test_bounded_inputs_and_strict_canonical_json(monkeypatch):
    import src.audit_bundle as bundle
    archive, _ = create_bundle({"data.json": {}})
    monkeypatch.setattr(bundle, "MAX_BYTES", 64)
    assert not verify_bundle(archive)["valid"]
    with pytest.raises(ValueError):
        create_bundle({"data.json": "x" * 65})
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({1: "coercion"})
    assert not verify_bundle(b"not a zip")["valid"]


def test_cli_verify_bundle_is_offline_and_returns_failure_exit(tmp_path):
    data, digest = create_bundle({"result.json": {"status": "verified_bounded"}})
    path = tmp_path / "audit.zip"
    path.write_bytes(data)
    command = [sys.executable, "-m", "src.cli", "verify-bundle", "--input", str(path)]
    good = subprocess.run(command + ["--expected-manifest-digest", digest], capture_output=True, text=True, check=False)
    assert good.returncode == 0
    assert json.loads(good.stdout)["valid"]
    bad = subprocess.run(command + ["--expected-manifest-digest", "0" * 64], capture_output=True, text=True, check=False)
    assert bad.returncode == 1


def test_cli_verification_bundle_records_request_result_and_runtime(tmp_path):
    request = tmp_path / "verify.json"
    request.write_text(json.dumps({"property": {"op": "true"}, "horizon": 2}))
    target = tmp_path / "evidence.zip"
    run = subprocess.run([sys.executable, "-m", "src.cli", "verify", "--input", str(request),
                          "--bundle", str(target), "--revision", "a" * 40],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert verify_bundle(target.read_bytes())["valid"]
    with zipfile.ZipFile(target) as archive:
        assert json.loads(archive.read("runtime.json"))["code_revision"] == "a" * 40
        assert json.loads(archive.read("result.json"))["horizon"] == 2
