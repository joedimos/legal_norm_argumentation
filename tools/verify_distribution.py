"""Exercise a built wheel in a fresh environment outside the source checkout.

Existing interpreter dependencies are reused without downloading anything. The
application itself must resolve inside the fresh environment. Source data bytes
are compared to the wheel before installed CLI and migration checks are run.
"""

import argparse
import hashlib
import json
import os
import site

# Launches only explicitly selected local build/test interpreters.
import subprocess  # nosec B404
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile


def run(command, *, cwd, env):
    # Trusted interpreter/argument lists, never a shell command.
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=180, check=True, shell=False).stdout  # nosec B603


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    root = Path(__file__).resolve().parents[1]
    checked = 0
    with ZipFile(wheel) as archive:
        for group in ("benchmarks", "schemas", "examples", "migrations"):
            for path in sorted((root / group).rglob("*")):
                if path.is_file() and path.suffix in {".json", ".py", ".mako", ".md"}:
                    require(archive.read("src/_data/" + path.relative_to(root).as_posix()) == path.read_bytes(), "resource bytes differ")
                    checked += 1
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLE_") and k not in {"PYTHONPATH", "PYTHONHOME"}}
    with tempfile.TemporaryDirectory(prefix="vle-wheel-check-") as directory:
        work = Path(directory)
        target = work / "environment"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(target)
        executable = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        packages = Path(run([str(executable), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], cwd=work, env=env).strip())
        # Reuse installed test dependencies, never the source package itself.
        (packages / "build_test_dependencies.pth").write_text("\n".join(site.getsitepackages()) + "\n")
        run([str(executable), "-m", "pip", "install", "--no-index", "--no-deps", "--ignore-installed", str(wheel)], cwd=work, env=env)
        bootstrap = (
            "import socket,sys,runpy,pathlib,src\n"
            "if not pathlib.Path(src.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()): raise RuntimeError('checkout import')\n"
            "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network forbidden')); "
            "sys.argv=sys.argv[1:]; runpy.run_module(sys.argv[0],run_name='__main__')"
        )
        commands = [
            ["src.cli", "benchmark"], ["src.cli", "experiment-benchmark"],
            ["src.cli", "family-benchmark"], ["src.cli", "faithfulness"],
        ]
        for command in commands:
            value = json.loads(run([str(executable), "-I", "-c", bootstrap, *command], cwd=work, env=env))
            require(value.get("all_passed", value.get("status") == "reported"), "installed CLI regression failed")
        contract = json.loads(run([str(executable), "-I", "-c", bootstrap, "src.api_contract"], cwd=work, env=env))
        require(contract["status"] == "compatible_with_baseline", "installed API contract differs")
        env["VLE_DATABASE_URL"] = "sqlite+pysqlite:///" + str(work / "installed.db")
        for options in ([], ["--check"]):
            result = json.loads(run([str(executable), "-I", "-c", bootstrap, "src.service.migrate", *options], cwd=work, env=env))
            require(result["passed"], "installed migrations failed")
        # The installed console scripts must be present, not just modules.
        scripts = target / ("Scripts" if os.name == "nt" else "bin")
        for name in ("vle-legato", "vle-legato-api", "vle-legato-migrate", "vle-legato-release-check", "vle-legato-importer", "vle-legato-api-contract"):
            require((scripts / (name + ".exe" if os.name == "nt" else name)).exists(), "console entry point is missing")
    print(json.dumps({"schema": "vle-distribution-check/v1", "passed": True, "resource_files": checked,
                      "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "network_calls": 0,
                      "decision_authorized": False, "legal_certification": False}, sort_keys=True))


if __name__ == "__main__":
    main()
