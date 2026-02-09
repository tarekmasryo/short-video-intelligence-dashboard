# Case Study — Short-Video Intelligence Dashboard

## Problem
Short-form exports vary a lot (different column names + missing fields). The goal was one fast dashboard that answers practical questions:
- Which creators/segments outperform consistently?
- What timing patterns (day/hour) correlate with stronger results?
- Which videos look like viral candidates under a clear definition?

## Approach
- Streamlit + Plotly for interactive, filter-first analysis.
- Best-effort column detection → normalize into stable concepts (views/likes/comments/shares/date + dimensions).
- Enrich features: time keys, engagement rates, duration buckets, performance tiers, virality labeling.

## Key Decisions
- Portable loading (upload + local fallback paths).
- Explainable virality score + quantile-based tiering to work across datasets of different scales.
- Caching + sampling to keep the UI responsive.
- Graceful degradation when key columns are missing.

## Results
A decision-ready dashboard with:
- Slice filters (platform/country/category/creator/date)
- Trends & timing views
- Creator/segment comparisons
- Virality lab (thresholding + candidates)
- Data explorer + filtered export

## Next Steps
- Configurable scoring weights
- Posting recommendations per segment
- Creator consistency metrics
- Exports to `artifacts/` + lightweight data validation
