# Deprecated Scripts

## `37_build_fixed_deep_learning_sample.legacy.py`

**Status:** DEPRECATED — do not run.

**Deprecated on:** 2026-08-09

**Reason:** Attempted to rebuild Step 30's exact 2M/500K rows and backfill categorical
features from upstream. This approach was cancelled as too fragile. Step 30 fixed
samples remain historical LightGBM/XGBoost/Logistic experiments only.

**Replacement:** `scripts/37_build_unified_modeling_sample.py`

Builds a new independent unified modeling sample at `data/modeling/unified_{train,valid}/`
for LightGBM, Wide & Deep, and DeepFM going forward.
