# Academic manuscript and reproducibility record

`manuscript.md` explains the complete project: source extraction and provenance,
formal interpretations, defeasible/argumentative reasoning, bounded temporal
verification, uncertainty and clarification, model revision, signed annotation,
production policy, evaluation and open research questions.

The manuscript describes the 0.9.0 runtime-code snapshot identified in
`evaluation.json`, with Phase 4/5 results, automated IP acquisition and strict RC
preparation. The paper update is a separate documentation commit. The previous
0.6.2 record remains in `evaluation-0.6.2.json`. Current code/resource hashes,
test counts and regression digests are retained. Expected labels and research
results remain synthetic/unreviewed; no independent legal-accuracy, live-provider,
live-model or production staging study is claimed. Negative/unavailable findings
and the remaining release actions are reported explicitly.

The PDF renderer uses ReportLab and DejaVu fonts, with native vector diagrams
and selectable text. Install the optional publication dependencies separately
from the service runtime:

```bash
python -m pip install -r docs/paper/requirements.txt
python docs/paper/render_paper.py --output output/pdf/legal_norm_argumentation_academic_paper.pdf
```

The default font directory is `/usr/share/fonts/truetype/dejavu`; use
`--font-dir` for another installation containing the named DejaVu files.
The renderer does not claim to implement a particular conference template.
Before submission, verify author information, venue format and policy,
independent study requirements and the current source/experiment record.

The user-facing PDF is delivered separately. It is not a service dependency or
an additional legal certification artifact.
