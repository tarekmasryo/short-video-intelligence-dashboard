# 📺 Short-Video Intelligence Dashboard — Creators, Virality & Segments

[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io/)<br>
[![Made with ❤️ by Tarek Masryo](https://img.shields.io/badge/Made%20by-Tarek%20Masryo-blue)](https://github.com/tarekmasryo)

An interactive **decision dashboard** for short-form video performance across platforms  
— built for **creators, analysts, and growth teams** who need **clear signals, not vanity charts**.

The app ingests any “**one row per video**” dataset, detects key fields automatically, and adds a **virality & engagement scoring layer** on top: tiers, thresholds, and segments you can actually act on.

---

## 🚀 Why this dashboard?

Most short-form analytics stop at:

> views, likes, a couple of charts, and a feeling that “something is working”.

This dashboard is designed to answer sharper questions:

- Which **creators, categories, and platforms** are really carrying performance?
- What does a **“viral” video** look like in *this* dataset, not in theory?
- How do **timing, duration, and country** affect reach and engagement?
- Which **segments** are under-performing and worth a reset?

It turns a CSV export into:

> **Growth charts, virality scores, creator leaderboards, and segment benchmarks** — all in one place.

---

## 🧩 Project Overview

| Component | Description |
| :-- | :-- |
| **Dashboard** | Streamlit app for short-video performance, virality scoring, and segment comparison. |
| **Input model** | Generic tabular data: one row per video, with best-effort auto-detection of core fields. |
| **Scoring layer** | Derived metrics: engagement rate, virality score, performance tiers, viral potential. |
| **Decisions** | Posting time strategy, creator & category bets, content mix, campaign reviews. |

---

## 📂 Expected Input Data

The app is data-agnostic, but works best when your dataset contains columns similar to:

| Concept | Example column names | Notes |
| :-- | :-- | :-- |
| **Views** | `views`, `view_count`, `views_count`, `play_count` | Numeric, per video. |
| **Likes** | `likes`, `like_count` | Optional but recommended. |
| **Comments** | `comments`, `comment_count` | Optional. |
| **Shares** | `shares`, `share_count` | Important for virality. |
| **Duration (sec)** | `duration_sec`, `video_length`, `length_sec` | Used to build duration buckets. |
| **Publish time** | `publish_date`, `published_at`, `upload_date`, `timestamp` | Parsed into date, week, month, hour, day-of-week. |
| **Platform** | `platform`, `source`, `app` | e.g. TikTok, YouTube, Instagram. |
| **Creator / Account** | `creator`, `channel`, `author`, `handle`, `username` | Used for the creator leaderboard. |
| **Category / Topic** | `category`, `topic`, `tag` | For content mix analysis. |
| **Country / Region** | `country`, `region`, `market`, `geo` | For segment comparison. |
| **Hashtags** | `hashtags`, `tags` | List/CSV/text; parsed into `hashtags_list`. |

If your naming is different (for example `video_views_total`), the app still tries to **auto-map** fields by pattern.

---

## 📈 Metrics & Scoring

The core of the app is a simple, transparent scoring layer — not a black-box model.

### Engagement metrics

```text
engagement_rate = (likes + comments + shares) / views
like_rate       = likes / views
comment_rate    = comments / views
share_rate      = shares / views
```

- Values are capped to avoid extreme outliers.
- Displayed as percentages across the dashboard.

### Virality Score (explained)

```text
virality_score =
  0.30 * log1p(views)
+ 0.40 * log1p(shares)
+ 0.30 * (engagement_rate * 100)
```

This gives you:

- A **single, comparable score** per video.
- A clean way to define “viral candidates” using a threshold slider.
- A quick feel for how **scale, spread, and engagement** interact.

### Performance Tiers

Built from view quantiles within the current filters:

- **Baseline** — up to median views  
- **Strong** — median to ~80th percentile  
- **Top 5%** — high performers  
- **Top 1%** — extreme outliers

These tiers drive:

- View distribution plots in the Overview tab.
- Quick-copy insights like “Top 5% videos cluster around X views”.

### Viral Potential

A simple label on top of the virality score:

- **High**, **Medium**, **Low**, or **Unknown**  
- Based on score quantiles (typically 75th and 90th percentile splits).

You can use it directly in:

- Reporting (“10% of filtered videos have High viral potential”).
- Filtering, segmenting, and reviewing candidate clips.

---

## 🧭 Dashboard Layout

The dashboard follows a **clean, tab-based structure** aligned with a production analytics workflow.

### 1️⃣ Overview — Executive Snapshot

- Hero KPIs for the current filter slice:
  - Filtered videos
  - Total views
  - Median engagement rate
  - Platforms in scope
- Secondary KPIs:
  - Rows after filters
  - Average views per video
  - Median duration (short-form window)
  - Share of videos with **High** viral potential
- Distribution views:
  - View distribution by **performance tier** (log-scale boxplot)
  - Viral potential breakdown (Low/Medium/High)

**Use it for:** one-slide summaries and quick health checks.

---

### 2️⃣ Growth & Timing — Trend & Posting Windows

- Time-series for:
  - `views`, `likes`, `comments`, `shares`, or `engagement_rate`
- Aggregation by:
  - **Day**, **Week**, or **Month**
- Dual view:
  - Total metric over time (line + area)
  - Period-over-period growth rate (%)
- KPIs:
  - Peak period
  - Average growth
  - Number of periods analysed
- Behaviour patterns:
  - Median views by **publish hour**
  - Median views by **day of week**

**Use it for:** planning posting schedules and understanding momentum.

---

### 3️⃣ Creators & Content — Leaderboard & Mix

- Rank creators by:
  - Total views
  - Median engagement rate
  - Total shares / comments
  - Median virality score
- Horizontal bar chart for **Top N creators**.
- Snapshot card:
  - Top 5 creators
  - Video count per creator
  - Metric values with clean formatting.
- Content mix:
  - Median views by **duration bucket**
  - Top categories by median views.

**Use it for:** creator deals, category bets, and content mix decisions.

---

### 4️⃣ Virality Lab — Thresholds & Candidates

- Virality score histogram with a live **threshold slider**.
- Viral candidate summary:
  - Count of videos above the threshold.
  - Share of filtered videos.
  - Median virality score by platform (when platform exists).
- Viral candidates table (top 100):
  - Title, platform, views, engagement rate, virality score, duration.
- Views vs engagement scatter plot (log views, sampled for big data).

**Use it for:** defining what “viral” means today, and reviewing the clips that meet that bar.

---

### 5️⃣ Segment Comparison — Benchmarks

- Segment by:
  - Platform, country, category, duration bucket, or performance tier.
- Metrics:
  - Views, likes, comments, shares, engagement rate, virality score.
- Aggregations:
  - Median, mean, or sum.
- Outputs:
  - Horizontal bar chart of top segments.
  - Card summarising the top 5 segments with human-readable numbers.

**Use it for:** performance benchmarking and spotting outlier segments.

---

### 6️⃣ Data Explorer — Under the Hood

- KPIs:
  - Rows (filtered)
  - Columns
  - Estimated memory (MB)
- Table preview (adjustable row count).
- **Download filtered data as CSV**.
- Optional:
  - Numeric summary (`describe`).
  - Column info (dtype, non-null/ null counts).

**Use it for:** data quality checks and exporting slices into notebooks or BI tools.

---

## 📸 Dashboard Preview

> Example screenshots — adapt paths to your own assets in `assets/`.

### Overview — Hero & KPIs

<p align="center">
  <img src="assets/short-video-overview.png" alt="Short-Video Intelligence Dashboard — overview hero with KPIs and performance tiers" />
</p>

---

### Growth & Timing — Monthly Comments Trend

<p align="center">
  <img src="assets/monthly-comments-growth.png" alt="Monthly comments trend with total comments and growth rate" />
</p>

---

### Creators & Content — Virality Leaderboard

<p align="center">
  <img src="assets/creators-virality-leaderboard.png" alt="Top creators ranked by virality score with snapshot of top performers" />
</p>

---

### Data Explorer — Filtered Dataset View

<p align="center">
  <img src="assets/data-explorer-view.png" alt="Data explorer with rows, columns, memory estimate, and preview table" />
</p>

---

## ⚙️ Quick Start

```bash
# Clone the repository
git clone https://github.com/TarekMasryo/short-video-intelligence-dashboard.git
cd short-video-intelligence-dashboard

# (Optional) create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

