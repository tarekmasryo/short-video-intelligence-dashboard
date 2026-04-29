# Changelog

## v0.1.3 (CI hardening)

- Updated GitHub Actions to current Node 24-compatible action versions.
- Added `fail-fast: false` to expose both Python matrix results during CI.
- Switched CI commands to `python -m ...` for more reliable tool resolution.
- Added a syntax-check step before tests.
- Replaced newer Streamlit `width="stretch"` calls with `use_container_width=True` for broader Streamlit compatibility.
- Removed generated cache artifacts from the release package.

## v0.1.2 (reviewed release)

- Hardened column detection to prefer creator identity fields over aggregate creator metrics.
- Added regression tests for creator, title, and category auto-detection.
- Kept the dashboard package layout and shipped demo dataset unchanged.

## v0.1.1 (publish-ready)

- README aligned to the actual package layout + added screenshots
- Lint clean under Ruff defaults (no unsafe fixes)
- CI checks formatting via `ruff format --check`
- Removed duplicate license file


## v0.1.0 (production refresh)

- Modularized the Streamlit app into a small package (`short_video_intel/`)
- Added unit tests (`pytest`) and CI workflow (lint + tests)
- Added Docker support (Dockerfile + .dockerignore)
- Added Streamlit config defaults and cleaned dependencies
- Kept the shipped sample dataset for out-of-the-box demo
