import pandas as pd

from src.process_dataset import run_pipeline


def test_dataset_pipeline_handles_rows_independently(tmp_path):
    source = tmp_path / "cases.csv"
    output = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "text": [
                "The work reflects the author's personality and free creative choices.",
                "No originality marker is present here.",
            ]
        }
    ).to_csv(source, index=False)
    result = run_pipeline(str(source), str(output), max_cases=None)
    assert result.loc[0, "arguments_count"] > 0
    assert result.loc[1, "arguments_count"] == 0
    assert output.exists()
