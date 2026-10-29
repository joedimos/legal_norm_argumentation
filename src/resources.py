"""Working-directory-independent access to shipped, versioned resources.

Explicit user input paths never pass through this resolver. Defaults refer to
this distribution, not a similarly named file in the caller's current directory.
"""

from pathlib import Path


def resource_path(group: str, name: str = "") -> Path:
    if group not in {"benchmarks", "schemas", "examples", "migrations"}:
        raise ValueError("unsupported resource group")
    if name and (Path(name).name != name or name in {".", ".."} or "\\" in name):
        raise ValueError("resource names must be flat")
    package = Path(__file__).resolve().parent
    for base in (package / "_data", package.parent):
        result = base / group / name if name else base / group
        if result.exists():
            return result
    raise FileNotFoundError("required distribution resource is missing")
