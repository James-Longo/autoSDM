# Tasks: Generalized Covariate Standardization

- [x] Implement Standardized Method (Random Forest)
    - [x] Implement `train_standardized_model` in `trainer.py`
    - [x] Implement automated nuisance optima detection
    - [x] Implement `generate_standardized_prediction_map` in `extrapolate.py`
- [x] Update CLI and R Wrappers
    - [x] Update `cli.py` with `--method` and `--nuisance-vars`
    - [x] Update `autoSDM.R`, `analyze_embeddings.R`, and `extrapolate.R`
- [x] Verification and Robustness
    - [x] Test with presence-only (Centroid) data
    - [x] Test with presence/absence (Standardized) data
    - [x] Verify with real Mountain Birdwatch data
    - [x] Implement automated missing value alerts
- [x] Documentation
    - [x] Update `walkthrough.md` with Standardized Mode instructions
