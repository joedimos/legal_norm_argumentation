"""Installed native-engine identity and adapter-contract fingerprints."""

from __future__ import annotations

import platform
from importlib.metadata import version
from pathlib import Path

from .pyreason_connector import BACKEND_VERSION, digest


def contract_sha256():
    root = Path(__file__).resolve().parent
    import hashlib

    names = [
        "pyreason_connector.py",
        "pyreason_worker.py",
        "pyreason_pool.py",
        "pyreason_program.py",
        "pyreason_normative.py",
        "pyreason_graph.py",
        "pyreason_runtime.py",
        "pyreason_qualification.py",
        "compliance.py",
        "defeasible.py",
    ]
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})


def native_identity(pr):
    import hashlib

    root = Path(pr.__file__).resolve().parent
    if Path(pr.__file__).name != "__init__.py" or version("pyreason") != BACKEND_VERSION:
        raise ValueError("stand-in or unsupported engine cannot qualify as native")
    names = [
        "__init__.py",
        "pyreason.py",
        "scripts/facts/fact.py",
        "scripts/rules/rule.py",
        "scripts/interpretation/interpretation.py",
        "scripts/utils/rule_parser.py",
    ]
    source = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    packages = {
        name: version(name) for name in ["pyreason", "numba", "llvmlite", "numpy", "networkx", "pandas", "PyYAML", "pytest", "memory-profiler", "psutil"]
    }
    result = {
        "provider": "native",
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "packages": packages,
        "engine_source_sha256": digest(source),
        "adapter_contract_sha256": contract_sha256(),
    }
    return {**result, "identity_sha256": digest(result)}


def installed_identity():
    """Fingerprint installed files without running PyReason import-time code."""
    from importlib.util import find_spec
    from types import SimpleNamespace

    spec = find_spec("pyreason")
    if spec is None or spec.origin is None:
        raise ValueError("native engine is not installed")
    return native_identity(SimpleNamespace(__file__=spec.origin))
