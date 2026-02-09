from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from short_video_intel.config import (
    APP_NAME,
    DEFAULT_MAIN_DATASET_FILENAME,
    DEFAULT_SAMPLE_LIMIT,
    DEFAULT_SEED,
    MAX_SCATTER_POINTS,
)
from short_video_intel.data import (
    detect_columns,
    enrich_data,
    label_viral_potential,
    load_csv_any,
    try_auto_load,
)
from short_video_intel.metrics import aggregate_time_metric
from short_video_intel.theme import DASHBOARD_CSS
from short_video_intel.ui import (
    format_creator_label,
    format_dim_label,
    format_metric_label,
    format_metric_value,
)

warnings.filterwarnings("ignore")


class DashboardApp:
    def __init__(self, app_dir: Path) -> None:
        self.app_dir = app_dir

    def run(self) -> None:
        st.set_page_config(
            page_title=APP_NAME,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        px.defaults.template = "plotly_dark"

        st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

        APP_DIR = self.app_dir

        default_main_paths = [
            APP_DIR / "data" / DEFAULT_MAIN_DATASET_FILENAME,
            APP_DIR / DEFAULT_MAIN_DATASET_FILENAME,
            APP_DIR / "short_video_dataset.csv",
            APP_DIR / "short_videos.csv",
            APP_DIR / "videos.csv",
            Path("/mnt/data") / DEFAULT_MAIN_DATASET_FILENAME,
        ]

        st.sidebar.markdown("### Data configuration")

        uploaded_main = st.sidebar.file_uploader(
            "Main short-video dataset (CSV)", type=["csv"], key="main_upload"
        )

        if uploaded_main is not None:
            raw = load_csv_any(uploaded_main)
            dataset_source = "Uploaded (sidebar)"
            dataset_detail = "User-provided file"
        else:
            raw, msg = try_auto_load(default_main_paths)
            dataset_source = "Default (auto-load)" if not raw.empty else "None"
            dataset_detail = msg

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Active dataset")
        st.sidebar.write(f"**Source:** {dataset_source}")
        st.sidebar.caption(dataset_detail)

        if dataset_source == "None" or raw.empty:
            st.error(
                "No main dataset loaded yet. Place a CSV in ./data/ or upload a file in the sidebar."
            )
            st.stop()

        if len(raw) > DEFAULT_SAMPLE_LIMIT:
            raw = raw.sample(DEFAULT_SAMPLE_LIMIT, random_state=DEFAULT_SEED)
            st.sidebar.info(f"Sampled to {DEFAULT_SAMPLE_LIMIT:,} rows for performance.")

        col_map = detect_columns(raw)
        df = enrich_data(raw, col_map)
        df = label_viral_potential(df)

        if df.empty:
            st.error("The file loaded but no usable rows remain after processing.")
            st.stop()

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Filters")

        filtered = df.copy()

        platform_filter = []
        creator_filter = []
        country_filter = []
        category_filter = []
        date_range = None
        min_views_filter = None

        if "platform" in df.columns:
            platforms_all = sorted(df["platform"].dropna().astype(str).unique().tolist())
            if len(platforms_all) > 0:
                default_platforms = platforms_all[: min(3, len(platforms_all))]
                platform_filter = st.sidebar.multiselect(
                    "Platform", platforms_all, default=default_platforms
                )
                if platform_filter:
                    filtered = filtered[filtered["platform"].isin(platform_filter)]

        if "country" in df.columns:
            countries_all = sorted(df["country"].dropna().astype(str).unique().tolist())
            default_countries = countries_all[:40]
            country_filter = st.sidebar.multiselect(
                "Country / Region", countries_all, default=default_countries
            )

        if "category" in df.columns:
            categories_all = sorted(df["category"].dropna().astype(str).unique().tolist())
            category_filter = st.sidebar.multiselect("Category", categories_all)

        if "creator" in df.columns:
            creators_all = sorted(df["creator"].dropna().astype(str).unique().tolist())
            creator_filter = st.sidebar.multiselect(
                "Creator (optional, for deep dives)",
                creators_all[:200],
            )

        if "date" in df.columns and df["date"].notna().any():
            d_min = df["date"].min()
            d_max = df["date"].max()
            date_range = st.sidebar.date_input(
                "Publish date range",
                value=(d_min, d_max),
                min_value=d_min,
                max_value=d_max,
            )

        if "views" in df.columns and df["views"].notna().any():
            min_views_filter = st.sidebar.number_input(
                "Minimum views (filter)",
                min_value=0,
                max_value=int(df["views"].max() or 0),
                value=0,
                step=1000,
            )

        if country_filter:
            filtered = filtered[filtered["country"].isin(country_filter)]

        if category_filter:
            filtered = filtered[filtered["category"].isin(category_filter)]

        if creator_filter:
            filtered = filtered[filtered["creator"].isin(creator_filter)]

        if date_range and "date" in filtered.columns:
            start, end = date_range
            filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

        if min_views_filter is not None and "views" in filtered.columns:
            filtered = filtered[filtered["views"] >= min_views_filter]

        fdf = filtered.copy()

        if fdf.empty:
            st.warning("Filters are too restrictive; no rows match the current selection.")
            st.stop()

        total_videos = len(fdf)
        total_views = fdf["views"].sum() if "views" in fdf.columns else 0.0
        median_engagement = (
            float(np.nanmedian(fdf["engagement_rate"])) * 100
            if "engagement_rate" in fdf.columns
            else 0.0
        )
        platform_text = (
            " · ".join(sorted(fdf["platform"].dropna().astype(str).unique().tolist()[:3]))
            if "platform" in fdf.columns
            else ""
        )

        hero_html = f"""
        <div class="hero">
          <h1>Short-Video Intelligence Dashboard</h1>
          <p class="hero-sub">
            Interactive analytics for short-form content performance across platforms — built for decision-making, not vanity charts.
          </p>
          <div class="hero-badges">
            <span class="hero-badge">One row per video</span>
            <span class="hero-badge">Automatic column detection</span>
            <span class="hero-badge">Virality score &amp; engagement tiers</span>
            <span class="hero-badge">Segment comparison &amp; benchmarks</span>
          </div>
          <div class="hero-metrics">
            <div class="hero-kpi">
              <div class="hero-kpi-label">Videos (filtered)</div>
              <div class="hero-kpi-value">{total_videos:,}</div>
            </div>
            <div class="hero-kpi">
              <div class="hero-kpi-label">Total views (filtered)</div>
              <div class="hero-kpi-value">{total_views:,.0f}</div>
            </div>
            <div class="hero-kpi">
              <div class="hero-kpi-label">Median engagement rate</div>
              <div class="hero-kpi-value">{median_engagement:.2f}%</div>
            </div>
            <div class="hero-kpi">
              <div class="hero-kpi-label">Platforms in scope</div>
              <div class="hero-kpi-value">{platform_text or "—"}</div>
            </div>
          </div>
        </div>
        """

        st.markdown(hero_html, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Rows in scope</div>
                  <div class="kpi-value">{len(fdf):,}</div>
                  <div class="kpi-sub">after filters</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            if "views" in fdf.columns:
                mean_views = float(np.nanmean(fdf["views"]))
                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <div class="kpi-label">Average views</div>
                      <div class="kpi-value">{mean_views:,.0f}</div>
                      <div class="kpi-sub">per video</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with k3:
            if "duration_min" in fdf.columns:
                med_dur = float(np.nanmedian(fdf["duration_min"]))
                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <div class="kpi-label">Median duration</div>
                      <div class="kpi-value">{med_dur:.2f} min</div>
                      <div class="kpi-sub">short-form window</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with k4:
            if "viral_potential" in fdf.columns:
                share_high = (fdf["viral_potential"] == "High").mean() * 100
                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <div class="kpi-label">High viral potential</div>
                      <div class="kpi-value">{share_high:.1f}%</div>
                      <div class="kpi-sub">of filtered videos</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("")

        show_data_tab = len(fdf.columns) <= 120

        tab_labels = [
            "Overview",
            "Growth & Timing",
            "Creators & Content",
            "Virality Lab",
            "Segment Comparison",
        ]
        if show_data_tab:
            tab_labels.append("Data Explorer")

        tabs = st.tabs(tab_labels)

        with tabs[0]:
            st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns([1.7, 1.3])

            with col_a:
                if "performance_tier" in fdf.columns and "views" in fdf.columns:
                    fig = px.box(
                        fdf,
                        x="performance_tier",
                        y="views",
                        color="performance_tier",
                        title="View distribution by performance tier",
                        log_y=True,
                    )
                    fig.update_layout(height=420, showlegend=False)
                    st.plotly_chart(fig, width="stretch")
                elif "views" in fdf.columns:
                    fig = px.histogram(
                        fdf,
                        x="views",
                        nbins=50,
                        title="View distribution",
                        log_y=True,
                    )
                    fig.update_layout(height=420)
                    st.plotly_chart(fig, width="stretch")

            with col_b:
                stats_rows = []

                if "views" in fdf.columns:
                    stats_rows.append(("Median views", f"{np.nanmedian(fdf['views']):,.0f}"))
                    stats_rows.append(
                        ("90th percentile views", f"{np.nanpercentile(fdf['views'], 90):,.0f}")
                    )

                if "engagement_rate" in fdf.columns:
                    stats_rows.append(
                        (
                            "Median engagement",
                            f"{(np.nanmedian(fdf['engagement_rate']) * 100):.2f}%",
                        )
                    )
                    stats_rows.append(
                        (
                            "Top 10% engagement threshold",
                            f"{(fdf['engagement_rate'].quantile(0.9) * 100):.2f}%",
                        )
                    )

                if "duration_min" in fdf.columns:
                    stats_rows.append(
                        ("Median duration (min)", f"{np.nanmedian(fdf['duration_min']):.2f}")
                    )

                if "viral_potential" in fdf.columns:
                    share_high = (fdf["viral_potential"] == "High").mean() * 100
                    stats_rows.append(("High viral potential share", f"{share_high:.1f}%"))

                if stats_rows:
                    html_rows = "".join(
                        f"<div style='display:flex;justify-content:space-between;font-size:13px;color:#e5e7eb;margin-bottom:4px;'><span style='color:#9ca3af;'>{k}</span><span style='font-weight:600;'>{v}</span></div>"
                        for k, v in stats_rows
                    )
                    card_html = f"""
                    <div class="card">
                      <div class="card-title">Quick snapshot</div>
                      {html_rows}
                    </div>
                    """
                else:
                    card_html = """
                    <div class="card">
                      <div class="card-title">Quick snapshot</div>
                      <div style="font-size:12px;color:#9ca3af;">No summary statistics detected for this dataset slice.</div>
                    </div>
                    """
                st.markdown(card_html, unsafe_allow_html=True)

            st.markdown("")
            st.markdown('<div class="section-title">Viral potential</div>', unsafe_allow_html=True)

            c1, c2 = st.columns([2, 1])

            with c1:
                if "viral_potential" in fdf.columns:
                    pie_fig = px.pie(
                        fdf,
                        names="viral_potential",
                        title="Viral potential distribution",
                    )
                    pie_fig.update_layout(height=380)
                    st.plotly_chart(pie_fig, width="stretch")
                else:
                    st.info(
                        "Viral potential is derived from virality score and engagement; it activates when those metrics exist."
                    )

            with c2:
                if "performance_tier" in fdf.columns and "engagement_rate" in fdf.columns:
                    tier_stats = (
                        fdf.groupby("performance_tier")["engagement_rate"]
                        .median()
                        .rename("median_eng")
                        .reset_index()
                        .sort_values("median_eng", ascending=False)
                    )
                    rows_html = ""
                    for row in tier_stats.itertuples(index=False):
                        rows_html += (
                            f"<div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;'>"
                            f"<span style='color:#9ca3af;'>{row.performance_tier}</span>"
                            f"<span style='font-weight:600;color:#e5e7eb;'>{row.median_eng*100:.2f}%</span>"
                            f"</div>"
                        )
                    card_html = f"""
                    <div class="card">
                      <div class="card-title">Tiers ranked by median engagement</div>
                      {rows_html}
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

        with tabs[1]:
            st.markdown('<div class="section-title">Growth & Timing</div>', unsafe_allow_html=True)

            available_metrics = [
                m
                for m in ["views", "likes", "comments", "shares", "engagement_rate"]
                if m in fdf.columns
            ]
            if not available_metrics:
                st.info("No time-series metrics detected in this dataset slice.")
            else:
                metric_choice = st.selectbox(
                    "Time-series metric",
                    available_metrics,
                    index=0,
                    key="trend_metric",
                )

                period_choice = st.radio(
                    "Time aggregation",
                    ["day", "week", "month"],
                    horizontal=True,
                )

                if "date" not in fdf.columns:
                    st.info("Time-series requires a valid publish date column.")
                else:
                    trends = aggregate_time_metric(fdf, metric=metric_choice, period=period_choice)
                    if trends.empty:
                        st.info("Time-series cannot be built from this dataset slice.")
                    else:
                        trends = trends.rename(columns={"value": "metric_value"}).copy()
                        trends = trends.sort_values("time", kind="stable")
                        trends["growth_rate"] = trends["metric_value"].pct_change() * 100.0

                        y_prefix = "Average"

                        fig = make_subplots(
                            rows=2,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.08,
                            row_heights=[0.7, 0.3],
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=trends["time"],
                                y=trends["metric_value"],
                                mode="lines+markers",
                                name=y_prefix,
                                fill="tozeroy",
                            ),
                            row=1,
                            col=1,
                        )

                        fig.add_trace(
                            go.Bar(
                                x=trends["time"],
                                y=trends["growth_rate"],
                                name="Growth rate (%)",
                                marker_color=[
                                    "#22c55e" if x >= 0 else "#ef4444"
                                    for x in trends["growth_rate"].fillna(0)
                                ],
                            ),
                            row=2,
                            col=1,
                        )

                        fig.update_layout(height=640, showlegend=True)
                        fig.update_yaxes(
                            title_text=f"{y_prefix} {format_metric_label(metric_choice)}",
                            row=1,
                            col=1,
                        )
                        fig.update_yaxes(title_text="Growth rate %", row=2, col=1)
                        st.plotly_chart(fig, width="stretch")

                        col_x, col_y, col_z = st.columns(3)

                        best_row = trends.loc[trends["metric_value"].idxmax()]
                        peak_period = best_row["time"]
                        peak_period_text = (
                            peak_period.strftime("%Y-%m-%d")
                            if hasattr(peak_period, "strftime")
                            else str(peak_period)
                        )
                        peak_value_text = format_metric_value(
                            metric_choice, best_row["metric_value"]
                        )
                        card_peak = f"""
                        <div class="card">
                          <div class="card-title">Peak period</div>
                          <div style="font-size:18px;font-weight:700;color:#e5e7eb;margin-top:4px;">{peak_period_text}</div>
                          <div style="font-size:12px;color:#94a3b8;margin-top:4px;">
                            {y_prefix} {format_metric_label(metric_choice)}: {peak_value_text}
                          </div>
                        </div>
                        """
                        avg_growth = trends["growth_rate"].mean()
                        avg_growth_value = 0.0 if pd.isna(avg_growth) else float(avg_growth)
                        card_avg = f"""
                        <div class="card">
                          <div class="card-title">Average growth</div>
                          <div style="font-size:18px;font-weight:700;color:{'#22c55e' if avg_growth_value >= 0 else '#ef4444'};margin-top:4px;">
                            {avg_growth_value:.1f}%
                          </div>
                          <div style="font-size:12px;color:#94a3b8;margin-top:4px;">
                            Across all observed periods
                          </div>
                        </div>
                        """
                        card_periods = f"""
                        <div class="card">
                          <div class="card-title">Periods analysed</div>
                          <div style="font-size:18px;font-weight:700;color:#e5e7eb;margin-top:4px;">{len(trends)}</div>
                          <div style="font-size:12px;color:#94a3b8;margin-top:4px;">
                            Based on non-empty points
                          </div>
                        </div>
                        """

                        with col_x:
                            st.markdown(card_peak, unsafe_allow_html=True)
                        with col_y:
                            st.markdown(card_avg, unsafe_allow_html=True)
                        with col_z:
                            st.markdown(card_periods, unsafe_allow_html=True)

            st.markdown("")
            st.markdown(
                '<div class="section-title">Posting time patterns</div>', unsafe_allow_html=True
            )

            c1, c2 = st.columns([1.4, 1.0])
            if "hour" in fdf.columns and "views" in fdf.columns:
                with c1:
                    hourly = fdf.groupby("hour")["views"].median().reset_index().sort_values("hour")
                    fig = px.bar(
                        hourly,
                        x="hour",
                        y="views",
                        title="Median views by publish hour",
                    )
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, width="stretch")

            if "day_of_week" in fdf.columns and "views" in fdf.columns:
                with c2:
                    dow = fdf.groupby("day_of_week")["views"].median().reset_index()
                    categories = [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]
                    dow["day_of_week"] = pd.Categorical(
                        dow["day_of_week"], categories=categories, ordered=True
                    )
                    dow = dow.sort_values("day_of_week")

                    fig = px.bar(
                        dow,
                        x="day_of_week",
                        y="views",
                        title="Median views by day of week",
                    )
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, width="stretch")

        with tabs[2]:
            st.markdown(
                '<div class="section-title">Creators & content</div>', unsafe_allow_html=True
            )

            if "creator" not in fdf.columns:
                st.info(
                    "No creator column detected. This view activates when a creator-like column exists in the data."
                )
            else:
                metric = st.selectbox(
                    "Rank creators by",
                    [
                        m
                        for m in [
                            "views",
                            "engagement_rate",
                            "shares",
                            "comments",
                            "virality_score",
                        ]
                        if m in fdf.columns
                    ],
                    index=0,
                )
                top_n = st.slider("Top N creators", min_value=5, max_value=50, value=15, step=1)

                agg = (
                    fdf.groupby("creator")
                    .agg(
                        views=("views", "sum") if "views" in fdf.columns else ("creator", "size"),
                        engagement_rate=("engagement_rate", "median")
                        if "engagement_rate" in fdf.columns
                        else ("creator", "size"),
                        shares=("shares", "sum")
                        if "shares" in fdf.columns
                        else ("creator", "size"),
                        comments=("comments", "sum")
                        if "comments" in fdf.columns
                        else ("creator", "size"),
                        virality_score=("virality_score", "median")
                        if "virality_score" in fdf.columns
                        else ("creator", "size"),
                        videos=("creator", "size"),
                    )
                    .reset_index()
                )

                agg = agg.sort_values(metric, ascending=False).head(top_n)

                c1, c2 = st.columns([1.8, 1.2])

                with c1:
                    fig = px.bar(
                        agg.sort_values(metric, ascending=True),
                        x=metric,
                        y="creator",
                        orientation="h",
                        title=f"Top {top_n} creators by {format_metric_label(metric)}",
                    )
                    fig.update_layout(height=520)
                    st.plotly_chart(fig, width="stretch")

                with c2:
                    rows_html = ""
                    metric_label = format_metric_label(metric)

                    for i, row in enumerate(agg.head(5).itertuples(index=False), start=1):
                        creator_label = format_creator_label(row.creator)
                        metric_value = format_metric_value(metric, getattr(row, metric))
                        videos_count = row.videos

                        rows_html += (
                            "<div style='margin-top:10px;'>"
                            f"<div style='font-size:12px;color:#9ca3af;'>#{i}</div>"
                            f"<div style='font-size:14px;font-weight:700;color:#e5e7eb;'>{creator_label}</div>"
                            f"<div style='font-size:12px;color:#94a3b8;'>"
                            f"Videos: {videos_count:,} · {metric_label}: {metric_value}"
                            "</div>"
                            "</div>"
                        )

                    card_html = (
                        "<div class='card'>"
                        "<div class='card-title'>Snapshot of top performers</div>"
                        f"{rows_html}"
                        "</div>"
                    )

                    st.markdown(card_html, unsafe_allow_html=True)

            st.markdown("")
            st.markdown('<div class="section-title">Content mix</div>', unsafe_allow_html=True)

            c1, c2 = st.columns([1.6, 1.4])

            with c1:
                if "duration_bucket" in fdf.columns and "views" in fdf.columns:
                    dur_agg = (
                        fdf.groupby("duration_bucket")["views"]
                        .median()
                        .rename("median_views")
                        .reset_index()
                    )
                    fig = px.bar(
                        dur_agg,
                        x="duration_bucket",
                        y="median_views",
                        title="Median views by duration bucket",
                    )
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, width="stretch")

            with c2:
                if "category" in fdf.columns and "views" in fdf.columns:
                    cat_agg = (
                        fdf.groupby("category")["views"]
                        .median()
                        .rename("median_views")
                        .reset_index()
                        .sort_values("median_views", ascending=False)
                        .head(12)
                    )
                    fig = px.bar(
                        cat_agg.sort_values("median_views", ascending=True),
                        x="median_views",
                        y="category",
                        orientation="h",
                        title="Top categories by median views",
                    )
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, width="stretch")

        with tabs[3]:
            st.markdown('<div class="section-title">Virality lab</div>', unsafe_allow_html=True)

            if "virality_score" not in fdf.columns:
                st.info(
                    "Virality score is computed when views, shares, and engagement can be derived."
                )
            else:
                scores = fdf["virality_score"].dropna()
                if scores.empty:
                    st.info("No valid virality scores after filters.")
                else:
                    q90 = scores.quantile(0.9)
                    min_thr = float(scores.min())
                    max_thr = float(scores.max())

                    virality_threshold = st.slider(
                        "Virality score threshold",
                        min_value=float(min_thr),
                        max_value=float(max_thr),
                        value=float(q90),
                    )

                    viral_candidates = fdf[fdf["virality_score"] >= virality_threshold].copy()
                    share_candidates = len(viral_candidates) / len(fdf) * 100

                    c1, c2 = st.columns([1.7, 1.3])
                    with c1:
                        fig = px.histogram(
                            fdf,
                            x="virality_score",
                            nbins=40,
                            title="Virality score distribution",
                        )
                        fig.add_vline(
                            x=virality_threshold,
                            line_color="#f97316",
                            line_dash="dash",
                            annotation_text="Threshold",
                            annotation_position="top right",
                        )
                        fig.update_layout(height=420)
                        st.plotly_chart(fig, width="stretch")

                    with c2:
                        extra_html = ""
                        if "platform" in fdf.columns and not viral_candidates.empty:
                            by_platform = (
                                viral_candidates.groupby("platform")["virality_score"]
                                .median()
                                .rename("median_vs")
                                .reset_index()
                                .sort_values("median_vs", ascending=False)
                            )
                            rows = ""
                            for row in by_platform.itertuples(index=False):
                                rows += (
                                    f"<div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;'>"
                                    f"<span style='color:#9ca3af;'>{row.platform}</span>"
                                    f"<span style='font-weight:600;color:#e5e7eb;'>{row.median_vs:.1f}</span>"
                                    f"</div>"
                                )
                            extra_html = (
                                "<hr style='border-color:rgba(31,41,55,0.9);margin:12px 0;'/>"
                                "<div style='font-size:13px;color:#e5e7eb;margin-bottom:6px;'>Platforms among viral candidates</div>"
                                + rows
                            )

                        card_html = f"""
                        <div class="card">
                          <div class="card-title">Viral candidates summary</div>
                          <div style="font-size:13px;color:#9ca3af;margin-bottom:6px;">
                            Videos above the current virality threshold.
                          </div>
                          <div style="font-size:28px;font-weight:800;color:#e5e7eb;margin-bottom:6px;">
                            {len(viral_candidates):,}
                          </div>
                          <div style="font-size:12px;color:#94a3b8;">
                            Share of dataset: {share_candidates:.1f}% of filtered videos.
                          </div>
                          {extra_html}
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

                    st.markdown("")
                    st.markdown(
                        '<div class="section-title">Viral candidates table</div>',
                        unsafe_allow_html=True,
                    )

                    if "title" in viral_candidates.columns:
                        preview_cols = ["title"]
                        for c in [
                            "platform",
                            "views",
                            "engagement_rate",
                            "virality_score",
                            "duration_min",
                        ]:
                            if c in viral_candidates.columns:
                                preview_cols.append(c)

                        st.dataframe(
                            viral_candidates[preview_cols].head(100),
                            width="stretch",
                            height=420,
                        )
                    else:
                        st.dataframe(
                            viral_candidates.head(100),
                            width="stretch",
                            height=420,
                        )

                    if "views" in fdf.columns and "engagement_rate" in fdf.columns:
                        st.markdown("")
                        st.markdown(
                            '<div class="section-title">Views vs engagement</div>',
                            unsafe_allow_html=True,
                        )

                        sample_base = (
                            fdf.sample(min(len(fdf), MAX_SCATTER_POINTS), random_state=DEFAULT_SEED)
                            if len(fdf) > MAX_SCATTER_POINTS
                            else fdf
                        )

                        sample_plot = sample_base[["views", "engagement_rate"]].copy()
                        sample_plot["views"] = pd.to_numeric(sample_plot["views"], errors="coerce")
                        sample_plot["engagement_rate"] = pd.to_numeric(
                            sample_plot["engagement_rate"], errors="coerce"
                        )
                        sample_plot = sample_plot.dropna(subset=["views", "engagement_rate"])

                        if sample_plot.empty:
                            st.info("No valid points to show for the current filters.")
                        else:
                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                    x=sample_plot["views"].astype(float).tolist(),
                                    y=sample_plot["engagement_rate"].astype(float).tolist(),
                                    mode="markers",
                                    name="Videos",
                                    marker=dict(size=5, opacity=0.45),
                                    hovertemplate="Views: %{x:.0f}<br>Engagement: %{y:.3f}<extra></extra>",
                                )
                            )
                            fig.update_layout(
                                title="Views vs engagement rate",
                                xaxis_title="Views (log scale)",
                                yaxis_title="Engagement rate",
                            )
                            fig.update_xaxes(type="log")
                            st.plotly_chart(fig, width="stretch")

        with tabs[4]:
            st.markdown(
                '<div class="section-title">Segment comparison</div>', unsafe_allow_html=True
            )

            candidate_dims = [
                c
                for c in ["platform", "country", "category", "duration_bucket", "performance_tier"]
                if c in fdf.columns
            ]
            candidate_metrics = [
                c
                for c in [
                    "views",
                    "likes",
                    "comments",
                    "shares",
                    "engagement_rate",
                    "virality_score",
                ]
                if c in fdf.columns
            ]

            if not candidate_dims or not candidate_metrics:
                st.info(
                    "This view activates when at least one categorical dimension and one numeric metric are present."
                )
            else:
                dim = st.selectbox("Segment by", candidate_dims, index=0)
                metric = st.selectbox("Metric", candidate_metrics, index=0)
                agg_type = st.radio("Aggregation", ["median", "mean", "sum"], horizontal=True)

                if agg_type == "median":
                    seg = fdf.groupby(dim)[metric].median().rename(metric).reset_index()
                elif agg_type == "mean":
                    seg = fdf.groupby(dim)[metric].mean().rename(metric).reset_index()
                else:
                    seg = fdf.groupby(dim)[metric].sum().rename(metric).reset_index()

                seg = seg.sort_values(metric, ascending=False)

                col1, col2 = st.columns([1.8, 1.2])

                with col1:
                    fig = px.bar(
                        seg.head(20).sort_values(metric, ascending=True),
                        x=metric,
                        y=dim,
                        orientation="h",
                        title=f"{format_metric_label(metric)} by {format_dim_label(dim)} ({agg_type})",
                    )
                    fig.update_layout(height=520)
                    st.plotly_chart(fig, width="stretch")

                with col2:
                    rows_html = ""
                    metric_label = format_metric_label(metric)
                    dim_label = format_dim_label(dim)

                    for i, row in enumerate(seg.head(5).itertuples(index=False), start=1):
                        seg_value = getattr(row, metric)
                        metric_value = format_metric_value(metric, seg_value)
                        dim_value = getattr(row, dim)

                        rows_html += (
                            "<div style='margin-top:10px;'>"
                            f"<div style='font-size:12px;color:#9ca3af;'>#{i} {dim_label}</div>"
                            f"<div style='font-size:15px;font-weight:700;color:#e5e7eb;'>{dim_value}</div>"
                            f"<div style='font-size:12px;color:#94a3b8;'>{metric_label}: {metric_value}</div>"
                            "</div>"
                        )

                    card_html = (
                        "<div class='card'>"
                        "<div class='card-title'>Top segments</div>"
                        f"{rows_html}"
                        "</div>"
                    )

                    st.markdown(card_html, unsafe_allow_html=True)

        if show_data_tab:
            with tabs[-1]:
                st.markdown(
                    '<div class="section-title">Data explorer</div>', unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Rows (filtered)", f"{len(fdf):,}")
                with c2:
                    st.metric("Columns", len(fdf.columns))
                with c3:
                    mem_mb = fdf.memory_usage(deep=True).sum() / 1024**2
                    st.metric("Estimated memory (MB)", f"{mem_mb:.1f}")

                rows_to_show = st.slider(
                    "Rows to display", min_value=20, max_value=1000, value=150, step=10
                )
                st.dataframe(
                    fdf.head(rows_to_show),
                    width="stretch",
                    height=480,
                )

                csv_bytes = fdf.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download filtered data as CSV",
                    data=csv_bytes,
                    file_name="short_video_filtered.csv",
                    mime="text/csv",
                )

                if st.checkbox("Show numeric summary"):
                    st.dataframe(fdf.describe(include="number"), width="stretch")

                if st.checkbox("Show column info"):
                    info_df = pd.DataFrame(
                        {
                            "column": fdf.columns,
                            "dtype": fdf.dtypes.values,
                            "non_null": fdf.count().values,
                            "null": fdf.isnull().sum().values,
                        }
                    )
                    st.dataframe(info_df, width="stretch")

        st.markdown(
            """
        <div class="footer">
          Short-Video Intelligence Dashboard · Built for creators, analysts, and growth teams who need
          clear signals, not noise.
        </div>
        """,
            unsafe_allow_html=True,
        )


def main() -> None:
    DashboardApp(Path(__file__).resolve().parent).run()


if __name__ == "__main__":
    main()
