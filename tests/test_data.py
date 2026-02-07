import pandas as pd

from short_video_intel.data import detect_columns, enrich_data, label_viral_potential


def test_detect_columns_basic():
    df = pd.DataFrame(
        {
            "view_count": [10, 20],
            "like_count": [1, 2],
            "comment_count": [0, 1],
            "share_count": [0, 1],
            "published_at": ["2025-01-01", "2025-01-02"],
            "platform": ["TikTok", "YouTube Shorts"],
            "channel_name": ["a", "b"],
            "hashtags": ["#x,#y", "#z"],
        }
    )
    m = detect_columns(df)
    assert m["views"] == "view_count"
    assert m["likes"] == "like_count"
    assert m["comments"] == "comment_count"
    assert m["shares"] == "share_count"
    assert m["date"] == "published_at"
    assert m["creator"] == "channel_name"


def test_enrich_and_label():
    raw = pd.DataFrame(
        {
            "views": [100, 200, 300, 0],
            "likes": [10, 20, 30, 0],
            "comments": [1, 2, 3, 0],
            "shares": [5, 5, 5, 0],
            "publish_date": ["2025-01-01", "2025-01-02", "2025-01-03", "bad"],
        }
    )
    col_map = detect_columns(raw)
    df = enrich_data(raw, col_map)
    assert "engagement_rate" in df.columns
    assert "virality_score" in df.columns
    df2 = label_viral_potential(df)
    assert "viral_potential" in df2.columns
