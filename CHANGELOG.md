# Changelog

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
