from __future__ import annotations

import pandas as pd


def aggregate_time_metric(df: pd.DataFrame, *, metric: str, period: str = "day") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["time", "value"])
    if metric not in df.columns:
        return pd.DataFrame(columns=["time", "value"])

    period = period.lower().strip()
    col = "date" if period == "day" else "week" if period == "week" else "month"
    if col not in df.columns:
        return pd.DataFrame(columns=["time", "value"])

    tmp = df[[col, metric]].copy()
    if period == "day":
        tmp[col] = pd.to_datetime(tmp[col], errors="coerce").dt.date
    tmp = tmp.dropna(subset=[col])
    out = tmp.groupby(col, as_index=False)[metric].mean(numeric_only=True)
    out = out.rename(columns={col: "time", metric: "value"})
    return out
