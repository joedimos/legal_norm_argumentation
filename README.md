# Legal Norm Argumentation — VLE/LEGATO Research Prototype

A research prototype for **faithful norm extraction, legal/normative argumentation, human review, and auditable verification**. The repository is organized around three connected problems:

1. **Extract and formalize norms faithfully** from laws, regulations, standards, policies, ethical principles, and other normative text.
2. **Reason with those norms** while keeping attacks, supports, conflicts, uncertainty, conditions, and exceptions explicit.
3. **Verify and audit the formalization/reasoning pipeline** rather than treating syntactic validity as evidence of legal or ethical correctness.

This is not a legal decision system, and extracted rules are not authoritative legal interpretations. Extraction output is intentionally represented as **reviewable candidates** with provenance and uncertainty.

## Implemented architecture

### 1. Faithful extraction and traceability

`src/norm_extraction.py` returns norm candidates with:

- stable candidate IDs;
- exact source spans and original sentences;
- deontic modality;
- actor and action;
- explicit conditions and exceptions;
- cross-reference metadata;
- ambiguity flags;
- confidence metadata and review status.

The extractor preserves the source sentence instead of silently rewriting it into a stronger logical claim. Optional transformer output is auxiliary evidence only.

### 2. Argumentation semantics

`src/argumentation.py` implements:

- conflict-free sets and defence;
- the Dung characteristic function;
- admissible, preferred, stable, and grounded semantics;
- grounded `in/out/undec` labels;
- self-attack/cycle handling;
- attack paths and influence analysis;
- separate support and attack graphs.

`LegalArgument` remains the primary representation. String IDs are accepted as a backwards-compatible shorthand.

### 3. Human review with invalidation

`src/review.py` binds each human review to a semantic fingerprint. If modality, actor, action, condition, exception, provenance, references, or ambiguity metadata changes, a prior approval becomes **stale** and must be reviewed again.

### 4. Verification / audit layer

`src/verification.py` checks:

- source-span integrity;
- missing actor/action;
- unknown modality;
- likely omitted conditions/exceptions;
- deontic source sentences omitted from the supplied candidate set.

The faithfulness score is a diagnostic signal, **not a certification result**.

### 5. Optional PyReason backend

`src/pyreason_connector.py` is isolated from the core so extraction, Dung semantics, human review, and auditing remain testable without PyReason. The adapter fails closed with a structured error if the installed PyReason/Python combination is unavailable or incompatible.

`requirements-pyreason.txt` pins the existing stable 3.7 line only for Python versions in its published compatibility range. PyReason compatibility should be validated independently before it is used for research results.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional; enables entity NER
```

For a compatible environment where the optional PyReason backend is required:

```bash
python -m pip install -r requirements-pyreason.txt
```

## CLI

```bash
python -m src.cli extract \
  --source-id example-policy \
  --text "If a breach is likely, the controller must notify the authority unless an exception applies."

python -m src.cli audit \
  --source-id example-policy \
  --text "If a breach is likely, the controller must notify the authority unless an exception applies."
```

## Python example

```python
from src.norm_extraction import extract_norms
from src.review import ReviewLedger
from src.verification import audit_extraction

text = "The operator must retain the audit log unless deletion is required by law."
norms = extract_norms(text, source_id="policy-7")
report = audit_extraction(text, norms)
assert not report["possible_omissions"]

ledger = ReviewLedger()
ledger.record(norms[0], "approved", reviewer="human-reviewer")
assert ledger.status(norms[0]) == "approved"
```

## Tests

```bash
python -m pytest -q
```

The suite includes an exhaustive invariant check over **all 512 directed attack graphs on three labeled arguments**, including self-attacks, plus focused extraction, originality, dataset-state, review, and verification tests.

## Research boundaries

The repository establishes a testable research foundation, not a complete legal-compliance certification system. Regex extraction does not fully resolve legal scope, incorporation by reference, temporal applicability, jurisdiction, authority hierarchy, or defeasibility. Conflict detection cannot by itself decide *lex specialis*, *lex superior*, or *lex posterior*. A reasoning engine can establish what follows from an encoding under its semantics; it does not establish that the encoding faithfully represents the governing norm.

**Verification of formalization precedes claims about verification of compliance.**
