# Short-Video Intelligence Dashboard

### Creator performance, virality scoring, timing patterns, and segment benchmarks

[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.11-2b5b84)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)
![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)
![CI](https://github.com/tarekmasryo/short-video-intelligence-dashboard/actions/workflows/ci.yml/badge.svg)

A production-style Streamlit dashboard that turns one-row-per-video datasets into actionable signals for creator performance, timing patterns, virality scoring, and segment benchmarks.

---

## Preview

![Overview](assets/short-video-overview.png)
![Creators & Virality](assets/creators-virality-leaderboard.png)
![Monthly Growth](assets/monthly-comments-growth.png)
![Data Explorer](assets/data-explorer-view.png)

---

## What this dashboard helps answer

- Which creators and segments are driving the strongest engagement?
- Which posting windows show better reach or interaction patterns?
- Which videos are likely breakout candidates based on configurable virality thresholds?
- How do platforms, countries, categories, and content types compare against each other?

---

## Key capabilities

- Auto-maps common fields: views, likes, comments, shares, publish time, platform, creator, category, country, and duration.
- Computes derived metrics: engagement rates, virality score, performance tiers, and viral potential.
- Provides decision views for:
  - Growth and timing trends across day, week, month, and posting windows.
  - Creator and content leaderboards with mix analysis.
  - Virality threshold review and candidate prioritization.
  - Segment benchmarking by platform, country, category, and other dimensions.
  - Data exploration with preview and filtered CSV export.

---

## Project structure

```text
.
├─ app.py
├─ short_video_intel/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ theme.py
│  └─ ui.py
├─ data/
│  └─ youtube_shorts_tiktok_trends_2025.csv
├─ assets/
│  ├─ short-video-overview.png
│  ├─ creators-virality-leaderboard.png
│  ├─ monthly-comments-growth.png
│  └─ data-explorer-view.png
├─ tests/
│  ├─ conftest.py
│  ├─ test_data.py
│  └─ test_metrics.py
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ .streamlit/
│  └─ config.toml
├─ CASE_STUDY.md
├─ DATASET_CARD.md
├─ CHANGELOG.md
├─ Dockerfile
├─ LICENSE
├─ requirements.txt
├─ requirements-dev.txt
└─ pyproject.toml
```

---

## Input data

The app works best when your dataset includes columns similar to:

| Concept | Example column names |
|---|---|
| Views | `views`, `view_count`, `play_count` |
| Likes | `likes`, `like_count` |
| Comments | `comments`, `comment_count` |
| Shares | `shares`, `share_count` |
| Duration (sec) | `duration`, `duration_sec`, `video_length` |
| Publish time | `publish_date`, `published_at`, `timestamp` |
| Platform | `platform`, `source`, `app` |
| Creator / Account | `creator`, `author`, `channel`, `username`, `handle` |
| Category / Topic | `category`, `topic`, `tag` |
| Country / Region | `country`, `region`, `market`, `geo` |
| Hashtags | `hashtags`, `tags`, `hashtag` |

Notes:

- Naming does not have to match exactly. The app uses best-effort auto-detection for common field names.
- Time fields are parsed into day, week, and month for trends, and into hour/day-of-week for posting-window analysis.
- For the bundled sample dataset, see `DATASET_CARD.md`.

---

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Quality checks

The GitHub Actions workflow runs a stable validation set:

```bash
python -m compileall -q app.py short_video_intel tests
python -m pytest -q
```

For local checks, install development dependencies first:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m compileall -q app.py short_video_intel tests
python -m pytest -q
```

---

## Docker

```bash
docker build -t short-video-intel .
docker run --rm -p 8501:8501 short-video-intel
```

Then open:

```text
http://localhost:8501
```

---

## Deployment notes

### Streamlit Community Cloud

- Main file: `app.py`
- Requirements: `requirements.txt`

### Hugging Face Spaces (Streamlit)

This repo is compatible with a Streamlit Space layout:

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`

---

## Engineering notes

- The app is organized around a thin Streamlit entrypoint and a small internal package for configuration, data loading, metric generation, theme helpers, and UI components.
- Tests cover core data and metric behavior used by the dashboard.
- CI validates Python importability and deterministic unit tests.
- Docker support is included for local containerized execution and deployment validation.

---

## License and data note

- Code: Apache 2.0. See `LICENSE`.
- Bundled sample data: included as a reproducible example for dashboard evaluation. See `DATASET_CARD.md`.
- Real platform exports: keep private unless you have verified privacy requirements, platform terms, and redistribution rights.
