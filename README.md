# SentinelRoute

Self-healing AIOps and adaptive MLOps deployment platform: a two-tier anomaly
model cascade, an MLflow registry with a promotion gate, a watchdog that heals
a failing service, and drift-triggered retraining.

> This is a temporary landing page. The full documentation (setup, demo
> walkthrough, API contract, metrics, and deployment notes) arrives with the
> documentation pull request.

## What is here

- `src/` — configuration, training, cascade API, watchdog, and drift check
- `tests/` — API, cascade, drift, and target-service tests
- `docker-compose.yml` — MLflow, API, and simulated target service
- `.github/workflows/ci.yml` — tests, GHCR publish, and manual retrain check

Start with `docker-compose.yml` and `src/config.py` until the full README lands.
