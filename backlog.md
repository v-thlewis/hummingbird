# Efficiency Backlog

## 2026-11-13 - re-fork note (read first)
- This fork was re-created from the parent project's `main`. Earlier runs analysed a stale non-default
  branch, so file paths/line numbers below may not match `main`. Re-verify each candidate against `main`.
- SVC vectorization: DONE and submitted upstream by the fork owner - see `upstreamed.md`. Do not re-implement.
  (On `main` the code lives in `hummingbird/ml/operator_converters/_sv_implementations.py`, not `sklearn/sv.py`.)
- `_utils.py` `*_installed()` caching: the earlier PR was LOST when the old fork was deleted; it was never
  submitted upstream. Re-verified 2026-11-13: `hummingbird/ml/_utils.py` on `main` still has no caching
  (no `lru_cache`). Top candidate to redo (Task 3), re-measure on `main`.

## History from previous fork (stale branch)
## Done
- [DONE] Code-Level HIGH: `hummingbird/ml/operator_converters/sklearn/sv.py` SVC.forward() had a
  per-predict-call Python double loop over class pairs (O(n_classes^2) x O(support vectors)) building
  tiny tensor slices/sums. Replaced with a precomputed coefficient matrix (built once at conversion time)
  + single matmul in forward(). Benchmark (2000 samples, 50 features, 8 classes, rbf kernel, 20 runs):
  baseline 227.7ms/predict -> optimized 42.4ms/predict (~5.4x). Predictions identical (verified via
  np.array_equal against sklearn and existing test suite, 20/20 SV tests pass). PR: efficiency/svc-vectorize-ovo-decision.

## Candidates (not yet started)
- `hummingbird/ml/operator_converters/_tree_implementations.py`: multiple `for i in range(self.num_trees)`
  / `for _ in range(self.max_tree_depth)` loops build Python lists at conversion time (one-time cost,
  lower priority than per-predict hot loops).
- `hummingbird/ml/operator_converters/onnx/tree_ensemble.py`: nested loops (lines ~69-160) building tree
  structures — appears to be conversion-time only (one-time), so LOW priority for predict-time energy,
  but worth checking if any is duplicated per predict call.
- `hummingbird/ml/operator_converters/_one_hot_encoder_implementations.py`: loops over `self.num_columns`
  — check whether this runs per-forward-call (could be MEDIUM if so).
- `_utils.py`: several `<lib>_installed()` functions do repeated `import x` inside try/except on every
  call rather than caching result — LOW impact (rarely called in hot paths) but easy caching win.
- No CI benchmark/perf regression job exists — Task 6 candidate: propose lightweight predict-time
  benchmark script + CI perf check (issue only, no infra changes without maintainer approval).

## Backlog cursor
Next run: continue scanning `_one_hot_encoder_implementations.py` and `_tree_implementations.py`
forward() methods for per-predict hot loops (Task 2/3), then consider Task 6 (benchmark infra proposal).

## Update 2026-09-03
- [DONE] Code-Level LOW->confirmed MEDIUM impact: `_utils.py` all nine `*_installed()` helpers
  (torch/onnx_ml_tools/onnx_runtime/sparkml/sklearn/lightgbm/xgboost/tvm/pandas/prophet) re-ran
  import machinery on every call. Added `functools.lru_cache(maxsize=None)` to each. Benchmark
  (200k calls): torch_installed 1.50s->0.008s (~194x), sklearn_installed 0.022s->0.008s (~2.7x),
  pandas_installed (not installed) 12.61s->0.008s (~1570x, failing imports are expensive).
  Full test suite: 37 failed/352 passed identical on main vs branch (pre-existing failures,
  verified via git stash). flake8 clean. PR: efficiency/cache-installed-checks.
- Remaining candidates unchanged: `_tree_implementations.py` conversion-time loops, 
  `onnx/tree_ensemble.py` nested loops, `_one_hot_encoder_implementations.py` forward() loops
  over num_columns (confirmed these DO run per-forward-call — worth investigating next: could
  vectorize the per-column eq/cat pattern, MEDIUM priority now upgraded from LOW).

## Backlog cursor
Next run: investigate `_one_hot_encoder_implementations.py` OneHotEncoder.forward() — loops over
num_columns building list of eq() results then torch.cat; confirmed this runs per predict call
(not conversion-time only). Consider vectorizing with a single batched comparison if categories
are same-length, or benchmark current cost. Then Task 6 (benchmark infra proposal, still not done).

## 2026-11-13 run (43521524680)
- SVC not yet on main; no re-measure possible.
- Measured `*_installed()` on main: 0-44us per call (failed import ~43us), conversion-time only. NEGLIGIBLE; DROP the lru_cache candidate (also risks breaking tests that mock imports).
