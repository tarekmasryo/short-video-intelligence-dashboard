from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

_F = TypeVar("_F", bound=Callable[..., Any])


def cache_data(func: _F) -> _F:
    """Cache heavy I/O steps when running inside Streamlit."""
    if st is None:
        return func
    return st.cache_data(show_spinner=False)(func)


@dataclass(frozen=True)
class ColumnMap:
    """Best-effort mapping from user columns → normalized concepts.

    All fields are optional because real-world datasets differ a lot.
    """

    views: str | None = None
    likes: str | None = None
    comments: str | None = None
    shares: str | None = None
    saves: str | None = None
    duration: str | None = None
    date: str | None = None
    platform: str | None = None
    creator: str | None = None
    title: str | None = None
    category: str | None = None
    country: str | None = None
    hashtags: str | None = None

    @classmethod
    def from_dict(cls, mapping: dict[str, str]) -> ColumnMap:
        keys = {
            "views",
            "likes",
            "comments",
            "shares",
            "saves",
            "duration",
            "date",
            "platform",
            "creator",
            "title",
            "category",
            "country",
            "hashtags",
        }
        filtered = {k: v for k, v in mapping.items() if k in keys}
        return cls(**filtered)

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ShortVideoDataset:
    """Small helper wrapper (optional) to keep df + mapped columns together."""

    df: pd.DataFrame
    columns: ColumnMap

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> ShortVideoDataset:
        mapping = detect_columns(df)
        return cls(df=df, columns=ColumnMap.from_dict(mapping))

    def enriched(self) -> ShortVideoDataset:
        enriched_df = enrich_data(self.df.copy(), self.columns.as_dict())
        return ShortVideoDataset(df=enriched_df, columns=self.columns)

    def labeled(self, *, q_mid: float = 0.75, q_high: float = 0.90) -> ShortVideoDataset:
        labeled_df = label_viral_potential(self.df.copy(), q_mid=q_mid, q_high=q_high)
        return ShortVideoDataset(df=labeled_df, columns=self.columns)


@cache_data
def load_csv_any(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(file, low_memory=False)
    except Exception:
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            return pd.read_csv(file, encoding="latin-1", low_memory=False)
        except Exception:
            return pd.DataFrame()


@cache_data
def load_csv_path(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1", low_memory=False)
        except Exception:
            return pd.DataFrame()


@cache_data
def try_auto_load(paths) -> tuple[pd.DataFrame, str]:
    for p in paths:
        try:
            pp = Path(p)
            if pp.exists() and pp.is_file():
                df_ = load_csv_path(pp)
                if not df_.empty:
                    return df_, f"Loaded default dataset: {pp.name}"
        except Exception:
            continue
    return pd.DataFrame(), "Default dataset not found. Upload a CSV to start."


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Detect common columns by name (exact match first, then substring match)."""
    if df is None or getattr(df, "empty", True):
        return {}

    column_map: dict[str, list[str]] = {
        "views": ["views", "view_count", "views_count", "total_views", "play_count"],
        "likes": ["likes", "like_count", "thumbs_up"],
        "comments": ["comments", "comment_count"],
        "shares": ["shares", "share_count"],
        "saves": ["saves", "save_count", "bookmarks"],
        "duration": ["duration_sec", "duration", "length_sec", "video_length", "video_length_sec"],
        "date": [
            "publish_date",
            "published_at",
            "publish_time",
            "date",
            "created_at",
            "upload_date",
            "timestamp",
        ],
        "platform": ["platform", "source", "app"],
        "creator": [
            "creator",
            "channel",
            "channel_name",
            "author",
            "uploader",
            "account",
            "handle",
            "username",
            "page",
        ],
        "title": ["title", "video_title", "name"],
        "category": ["category", "topic", "tag"],
        "country": ["country", "region", "market", "geo"],
        "hashtags": ["hashtags", "tags", "hashtag"],
    }

    detected: dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}

    # Exact match
    for key, candidates in column_map.items():
        for cand in candidates:
            if cand in lower_cols:
                detected[key] = lower_cols[cand]
                break

    # Substring match (fallback)
    for key, candidates in column_map.items():
        if key in detected:
            continue
        for cand in candidates:
            for c in df.columns:
                if cand in c.lower():
                    detected[key] = c
                    break
            if key in detected:
                break

    return detected


def parse_hashtags_cell(x) -> list[str]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list | tuple):
                return [str(t).strip() for t in v if str(t).strip()]
        except Exception:
            pass
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    if "#" in s and " " in s:
        return [t for t in s.split() if t.startswith("#")]
    return [s]


def enrich_data(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Create normalized columns + derived metrics (engagement, tiers, virality score)."""
    if df is None or df.empty:
        return df

    df = df.copy()

    for key in ["views", "likes", "comments", "shares", "saves", "duration"]:
        src = col_map.get(key)
        if src and src in df.columns:
            df[key] = pd.to_numeric(df[src], errors="coerce")

    for key in ["platform", "creator", "title", "category", "country"]:
        src = col_map.get(key)
        if src and src in df.columns:
            df[key] = df[src].astype(str)

    date_src = col_map.get("date")
    if date_src and date_src in df.columns:
        dt = pd.to_datetime(df[date_src], errors="coerce")
        df["publish_ts"] = dt
        df["date"] = df["publish_ts"].dt.date
        df["month"] = df["publish_ts"].dt.to_period("M").astype(str)
        df["week"] = df["publish_ts"].dt.to_period("W").astype(str)
        df["day_of_week"] = df["publish_ts"].dt.day_name()
        df["hour"] = df["publish_ts"].dt.hour

    if "duration" in df.columns:
        df["duration_min"] = df["duration"] / 60.0

    if all(c in df.columns for c in ["views", "likes", "comments", "shares"]):
        views_safe = df["views"].replace(0, np.nan)
        total_eng = df[["likes", "comments", "shares"]].sum(axis=1)
        df["engagement_rate"] = (total_eng / views_safe).clip(upper=10)
        df["like_rate"] = (df["likes"] / views_safe).clip(upper=1)
        df["comment_rate"] = (df["comments"] / views_safe).clip(upper=1)
        df["share_rate"] = (df["shares"] / views_safe).clip(upper=1)

        df["virality_score"] = (
            np.log1p(df["views"]).fillna(0) * 0.30
            + np.log1p(df["shares"]).fillna(0) * 0.40
            + (df["engagement_rate"].fillna(0) * 100) * 0.30
        )

    if "duration_min" in df.columns:
        df["duration_bucket"] = pd.cut(
            df["duration_min"],
            bins=[0, 0.15, 0.3, 0.6, 1.0, 2.0, np.inf],
            labels=["<9s", "9–18s", "18–36s", "36–60s", "1–2min", "2min+"],
            include_lowest=True,
        )

    if "views" in df.columns:
        views = df["views"].replace(0, np.nan)
        if views.notna().any():
            q50 = views.quantile(0.5)
            q80 = views.quantile(0.8)
            q95 = views.quantile(0.95)

            def bucket(v):
                if pd.isna(v):
                    return "Unknown"
                if v <= q50:
                    return "Baseline"
                if v <= q80:
                    return "Strong"
                if v <= q95:
                    return "Top 5%"
                return "Top 1%"

            df["performance_tier"] = views.map(bucket)

    hash_src = col_map.get("hashtags")
    if hash_src and hash_src in df.columns:
        df["hashtags_list"] = df[hash_src].apply(parse_hashtags_cell)

    return df


def label_viral_potential(
    df: pd.DataFrame, *, q_mid: float = 0.75, q_high: float = 0.90
) -> pd.DataFrame:
    """Assign a coarse 'viral_potential' label based on virality_score quantiles."""
    if df is None or df.empty or "virality_score" not in df.columns:
        return df

    scores = df["virality_score"].dropna()
    if scores.empty:
        return df

    q_mid = float(np.clip(q_mid, 0.01, 0.99))
    q_high = float(np.clip(q_high, 0.01, 0.99))
    if q_high <= q_mid:
        q_high = min(0.99, q_mid + 0.05)

    mid = scores.quantile(q_mid)
    high = scores.quantile(q_high)

    def label(v):
        if pd.isna(v):
            return "Unknown"
        if v >= high:
            return "High"
        if v >= mid:
            return "Medium"
        return "Low"

    df["viral_potential"] = df["virality_score"].map(label)
    return df
