from __future__ import annotations

from html import escape as _html_escape

import numpy as np
import streamlit as st


def escape_html(value, *, default: str = "", max_length: int | None = None) -> str:
    """Return a safe string for custom HTML fragments rendered by Streamlit."""
    if value is None:
        text = default
    elif isinstance(value, (np.floating, float)) and np.isnan(value):
        text = default
    else:
        text = str(value)

    if max_length is not None:
        text = text[:max_length]
    return _html_escape(text, quote=True)


def format_metric_label(metric: str) -> str:
    return metric.replace("_", " ").title()


def format_metric_value(metric: str, value) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except Exception:
        return escape_html(value)
    if metric == "engagement_rate":
        return f"{v * 100:.2f}%"
    if metric in ["views", "likes", "comments", "shares"]:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def format_dim_label(dim: str) -> str:
    return dim.replace("_", " ").title()


def format_creator_label(raw) -> str:
    if raw is None:
        return "Unknown"
    if isinstance(raw, (np.floating, float)):
        if np.isnan(raw):
            return "Unknown"
        v = float(raw)
        if v.is_integer():
            return f"Creator {int(v):,}"
        return f"Creator {v:,.0f}"
    if isinstance(raw, (np.integer, int)):
        return f"Creator {int(raw):,}"
    return escape_html(raw, default="Unknown", max_length=40)


def warn_no_rows() -> None:
    st.warning("No rows match this filter combination.")
