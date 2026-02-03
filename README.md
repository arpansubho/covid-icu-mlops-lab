# MLOps Mini Project – COVID Mortality Prediction

## Project Overview
This project builds an end-to-end machine learning system to predict COVID patient mortality using structured medical data.

We demonstrate:
- Data cleaning & preprocessing
- Model training (baseline + stronger model)
- Experiment tracking with MLflow
- API serving with FastAPI
- Dockerized inference
- CI with tests

## Target
Binary classification:
- 1 = patient died  
- 0 = patient survived  

Derived from `DATE_DIED`.

## Instructor Notes

### Common Errors
- GitHub push rejected due to large files → use `.gitignore` for `data/`, `models/`, `mlruns/`
- CI failures due to missing packages → always sync `requirements.txt`
- Single-class training crash → handled in `train.py`

### Extensions
- Add DVC for dataset versioning
- Replace drift stub with Evidently or PSI
- Add model registry (MLflow DB backend)
- Deploy on cloud (EC2 / Render / Fly.io)

### Architecture Summary
- Training: sklearn + MLflow
- Serving: FastAPI + Docker
- CI: GitHub Actions
- Monitoring: logging + drift stub

