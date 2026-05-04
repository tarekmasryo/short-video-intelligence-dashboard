# Changelog

## v0.1.5 (final publishing polish)

- Corrected performance-tier labels so quantile names match the underlying thresholds.
- Added a dataset card that documents the bundled sample data scope and usage limits.
- Escaped dataset-derived text before rendering custom HTML cards.
- Added Docker build validation to CI alongside syntax checks and unit tests.
- Streamlined release notes to describe CI behavior without weakening the quality signal.

## v0.1.4 (stable GitHub CI)

- Simplified CI to a single Python 3.11 validation job for predictable green builds.
- Updated workflow actions to Node 24-compatible action versions.
- Streamlined CI around deterministic runtime validation and unit tests.
- Pinned runtime and test dependencies for reproducible installs.
- Added `.python-version` to keep local and CI Python versions aligned.

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
- Kept the dashboard package layout and shipped sample dataset unchanged.

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
- Kept the shipped sample dataset for out-of-the-box sample run
