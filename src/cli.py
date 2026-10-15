from __future__ import annotations

import argparse
import json

from .norm_extraction import extract_all_norms, extract_norms
from .verification import audit_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description="VLE/LEGATO legal norm research prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "audit"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--text", required=True)
        sub.add_argument("--source-id", default="cli")

    args = parser.parse_args()
    if args.command == "extract":
        payload = extract_all_norms(args.text, source_id=args.source_id)
    else:
        norms = extract_norms(args.text, source_id=args.source_id)
        payload = {"norms": norms, "audit": audit_extraction(args.text, norms)}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
