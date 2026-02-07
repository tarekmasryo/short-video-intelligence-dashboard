from __future__ import annotations

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
    --accent-gradient: linear-gradient(135deg, #22c55e 0%, #14b8a6 50%, #0ea5e9 100%);
    --bg-elevated: radial-gradient(circle at top left, rgba(15,23,42,1), rgba(15,23,42,0.94));
    --bg-card: rgba(15,23,42,0.96);
    --border-subtle: rgba(148,163,184,0.35);
    --text-muted: #9ca3af;
    --chip-bg: rgba(15,23,42,0.96);
    --chip-border: rgba(148,163,184,0.4);
}

html, body, .stApp {
    background: radial-gradient(circle at top, #020617 0%, #020617 35%, #020617 100%);
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
.hero {
    padding: 32px 32px 28px 32px;
    border-radius: 24px;
    background: radial-gradient(circle at top left, rgba(79, 70, 229, 0.3), transparent 60%),
                radial-gradient(circle at bottom right, rgba(236, 72, 153, 0.25), transparent 55%),
                rgba(15, 23, 42, 0.96);
    border: 1px solid rgba(79, 70, 229, 0.55);
    box-shadow: 0 24px 80px rgba(15, 23, 42, 0.8);
    position: relative;
    overflow: hidden;
}

.hero h1 {
    margin: 0 0 8px 0;
    font-size: 34px;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    margin: 4px 0 18px 0;
    font-size: 14px;
    color: #e5e7eb;
    max-width: 780px;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 20px;
}

.hero-badge {
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.55);
    color: #cbd5f5;
    background: rgba(15,23,42,0.92);
}
.hero-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-top: 6px;
}

.hero-kpi {
    padding: 14px 16px;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.96);
    border: 1px solid rgba(148, 163, 184, 0.35);
}

.hero-kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 6px;
}

.hero-kpi-value {
    font-size: 20px;
    font-weight: 700;
    color: #e5e7eb;
}
.kpi-row {
    margin-top: 18px;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}

.kpi-card {
    padding: 16px 18px;
    border-radius: 18px;
    background: radial-gradient(circle at top left, rgba(15,23,42,0.98), rgba(15,23,42,0.96));
    border: 1px solid rgba(148, 163, 184, 0.3);
}

.kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
}

.kpi-value {
    font-size: 18px;
    font-weight: 700;
    margin-top: 6px;
    color: #e5e7eb;
}

.kpi-sub {
    margin-top: 4px;
    font-size: 11px;
    color: #6b7280;
}
.card {
    padding: 18px 18px 16px 18px;
    border-radius: 18px;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    min-height: 170px;
}

.card-soft {
    padding: 18px 18px 16px 18px;
    border-radius: 18px;
    background: radial-gradient(circle at top left, rgba(15,23,42,0.96), rgba(15,23,42,0.96));
    border: 1px dashed rgba(75,85,99,0.9);
}

.card-title {
    font-size: 14px;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 10px;
}
.section-title {
    margin: 4px 0 14px 0;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #e5e7eb;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    position: relative;
    padding-left: 14px;
}

.section-title::before {
    content: "";
    position: absolute;
    left: 0;
    top: 52%;
    transform: translateY(-50%);
    width: 3px;
    height: 18px;
    border-radius: 999px;
    background: linear-gradient(120deg, #6366f1, #f97316);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    padding: 12px;
    margin-top: 24px;
    margin-bottom: 6px;
    background: radial-gradient(circle at top left, rgba(30,64,175,0.45), rgba(15,23,42,0.98));
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.35);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px !important;
    padding: 10px 24px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #9ca3af !important;
    background: transparent !important;
    border: 0 !important;
}

button[data-baseweb="tab"]:hover {
    background: rgba(30, 64, 175, 0.25) !important;
    color: #e5e7eb !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--primary-gradient) !important;
    color: #f9fafb !important;
    box-shadow: 0 14px 35px rgba(79, 70, 229, 0.55);
}
[data-testid="stSidebar"] {
    background: radial-gradient(circle at top left, #020617, #020617);
    border-right: 1px solid rgba(31,41,55,0.9);
}

[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {
    font-size: 12px;
    color: #e5e7eb;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.footer {
    margin-top: 40px;
    padding: 18px 0 10px 0;
    text-align: center;
    color: #6b7280;
    font-size: 11px;
    border-top: 1px solid rgba(31,41,55,0.9);
}
</style>
"""
