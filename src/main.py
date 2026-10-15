from __future__ import annotations

from .argumentation import LegalArgument, OptimizedAAF
from .norm_extraction import extract_norms
from .verification import audit_extraction


def main() -> None:
    text = "If a breach is likely, the controller must notify the authority unless an exception applies."
    norms = extract_norms(text, source_id="demo")
    print("Extracted norms:", norms)
    print("Audit:", audit_extraction(text, norms))

    arguments = [
        LegalArgument("notify", "A qualifying breach occurred", "Notification is required"),
        LegalArgument("exception", "A valid exception applies", "Notification is not required"),
    ]
    aaf = OptimizedAAF(arguments, attacks={("exception", "notify")})
    print("Grounded labeling:", aaf.grounded_labeling())


if __name__ == "__main__":
    main()
