# Academic manuscript and reproducibility record

`manuscript.md` explains the complete project: source extraction and provenance,
formal interpretations, defeasible/argumentative reasoning, bounded temporal
verification, uncertainty and clarification, model revision, signed annotation,
production policy, evaluation and open research questions.

The manuscript describes software 0.6.2 at code commit
`9dbb08289ca6a494f5f514332e5201c529a74bdf`. The public code tree matches the
locally tested implementation. `evaluation.json` records local test counts and
regression corpus digests. Expected regression labels are synthetic; no
independent legal-accuracy or live-model study is claimed.

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
