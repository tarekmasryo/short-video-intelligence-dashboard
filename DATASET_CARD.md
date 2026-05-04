# Dataset Card — Short-Video Trends Sample

## Summary

This repository includes a sample short-video analytics dataset for reproducible dashboard evaluation. The dataset is structured as one row per video and is intended to exercise the dashboard workflow end to end: field detection, metric enrichment, growth analysis, creator ranking, virality scoring, segment comparison, and filtered export.

## File

```text
data/youtube_shorts_tiktok_trends_2025.csv
```

## Intended use

Use this sample to run the Streamlit app locally, validate dashboard behavior, reproduce screenshots, and test analytics logic before connecting private or production exports.

## Data characteristics

The app expects video-level records with common fields such as views, likes, comments, shares, publish date, platform, creator/account, category/topic, country/region, duration, and hashtags. Column names do not need to match exactly because the app performs best-effort field detection.

## Important limitations

- This sample is provided for dashboard evaluation and portfolio reproducibility.
- It should not be treated as an official platform dataset or a source of market truth.
- If you replace the sample with real exports, verify privacy, platform terms, and redistribution rights before publishing the data.

## License note

Code in this repository is licensed under Apache 2.0. The bundled sample data is included only as a reproducible example for this dashboard package. Keep private platform exports out of public repositories unless you have explicit rights to share them.
