# 📺 Short-Video Intelligence Dashboard 🎬
### Creators 👤 • Virality 🚀 • Timing ⏱️ • Segments 🧩 (Decision-Ready Analytics ✅)

[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.11%2B-2b5b84)
![Ruff](https://img.shields.io/badge/lint-ruff-261230)
![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)

A production-style **Streamlit** dashboard that converts “one row per video” datasets into actionable signals: **creator performance**, **timing patterns**, **virality scoring**, and **segment benchmarks**.

---

## ✨ Key capabilities

- 🧠 **Auto-maps common fields** (views, likes, comments, shares, publish time, platform, creator, category, country, duration).
- 🧮 Computes **derived metrics** (engagement rates, virality score, performance tiers, viral potential).
- 🧭 Provides decision views for:
  - 📈 **Growth & timing** (day/week/month trends + posting windows)
  - 🏆 **Creators & content** (leaderboards + mix)
  - 🧪 **Virality lab** (threshold + candidates)
  - 🧩 **Segment comparison** (benchmarks by platform/country/category/etc.)
  - 🔎 **Data explorer** (preview + export filtered CSV)

---

## 🧱 Project structure

```text
.
├─ app.py
├─ short_video/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ ui.py
│  ├─ viz.py
│  └─ theme.py
├─ tests/
│  ├─ test_data.py
│  └─ test_metrics.py
├─ .streamlit/
│  └─ config.toml
├─ requirements.txt
├─ requirements-dev.txt
├─ pyproject.toml
└─ README.md
```

---

## 🧾 Input data

The app is data-agnostic but works best when your dataset contains columns similar to:

| Concept | Example column names |
|---|---|
| Views 👀 | `views`, `view_count`, `play_count` |
| Likes 👍 | `likes`, `like_count` |
| Comments 💬 | `comments`, `comment_count` |
| Shares 🔁 | `shares`, `share_count` |
| Duration (sec) ⏳ | `duration`, `duration_sec`, `video_length` |
| Publish time 🗓️ | `publish_date`, `published_at`, `upload_date`, `timestamp` |
| Platform 📱 | `platform`, `source`, `app` |
| Creator / Account 👤 | `creator`, `author`, `channel`, `username`, `handle` |
| Category / Topic 🏷️ | `category`, `topic`, `tag` |
| Country / Region 🌍 | `country`, `region`, `market`, `geo` |
| Hashtags #️⃣ | `hashtags`, `tags` |

Notes:
- 🧩 If your naming differs (e.g., `video_views_total`), the app attempts best-effort **auto-mapping**.
- ⏱️ Time fields are parsed into **day/week/month** keys for trends and into **hour/day-of-week** keys for timing patterns.

---

## 📐 Metrics & scoring (transparent)

### 📊 Engagement metrics

```text
engagement_rate = (likes + comments + shares) / views
like_rate       = likes / views
comment_rate    = comments / views
share_rate      = shares / views
```

### 🚀 Virality score

```text
virality_score =
  0.30 * log1p(views)
+ 0.40 * log1p(shares)
+ 0.30 * (engagement_rate * 100)
```

### 🏷️ Performance tiers

Tiers are computed from view quantiles within the active filters:
- Baseline (≤ median)
- Strong (median → ~80th percentile)
- Top 5%
- Top 1%

---

## 🧭 Dashboard tabs

### 1) 🧾 Overview
Executive KPIs, tier distributions, viral potential breakdown.  
Use for quick health checks and summary views.

### 2) 📈 Growth & Timing
Day/week/month trends, growth rate, hour/day-of-week patterns.  
Use for scheduling strategy and momentum analysis.

### 3) 🏆 Creators & Content
Creator leaderboards by selected metric + duration/category mix.  
Use for creator bets, content planning, and category focus.

### 4) 🧪 Virality Lab
Interactive threshold, viral candidates table, and scatter sampling.  
Use for defining “viral” for the current dataset and reviewing candidates.

### 5) 🧩 Segment Comparison
Benchmarks across platform/country/category/duration bucket/tier.  
Use for diagnosing under/over-performing segments.

### 6) 🔎 Data Explorer
Preview, memory estimate, and export filtered CSV.  
Use for quality checks and exporting slices.

---

## ⚙️ Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## ✅ Quality checks (recommended)

```bash
pip install -r requirements-dev.txt

ruff check .
ruff format . --check
pytest -q
```

---

## 🚀 Deployment notes

### ☁️ Streamlit Community Cloud
- Main file: `app.py`
- Requirements: `requirements.txt`

### 🖧 Local network access
When running locally, Streamlit prints a **Network URL** for access from another device on the same LAN.

---

## 🧯 Troubleshooting

### KeyError related to publish time (e.g., `date`) 🗓️
The dataset likely does not include a recognizable publish time column or it contains non-parseable values.
- Ensure a publish time column exists (e.g., `publish_date`, `published_at`, `timestamp`).
- Prefer ISO timestamps or standard date formats.

### Ruff modified files locally 🧹
If you run `ruff check . --fix` or `ruff format .`, commit the changes so the deployed version matches what you tested.

---

## 📄 License
See `LICENSE` (if included).
