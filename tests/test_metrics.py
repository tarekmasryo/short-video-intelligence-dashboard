import pandas as pd

from short_video_intel.metrics import aggregate_time_metric


def test_aggregate_time_metric_daily() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
            "week": ["2025-W01", "2025-W01", "2025-W01"],
            "month": ["2025-01", "2025-01", "2025-01"],
            "views": [10, 30, 60],
        }
    )
    out = aggregate_time_metric(df, metric="views", period="day")
    assert list(out.columns) == ["time", "value"]
    assert len(out) == 2
    assert out["value"].iloc[0] == 20
    assert out["value"].iloc[1] == 60


def test_aggregate_time_metric_weekly() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-08"]),
            "week": ["2025-W01", "2025-W02"],
            "month": ["2025-01", "2025-01"],
            "views": [10, 90],
        }
    )
    out = aggregate_time_metric(df, metric="views", period="week")
    assert len(out) == 2
    assert out["time"].tolist() == ["2025-W01", "2025-W02"]
