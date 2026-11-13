# Changes Submitted to the Parent Project

Maintained by the fork owner. Never delete entries. Do not record parent-project issue/PR numbers or links here.

## SVC one-vs-one decision vectorization
- Status: submitted - awaiting merge (submitted 2026-11-13)
- What: `SVC.forward()` in `hummingbird/ml/operator_converters/_sv_implementations.py` now uses a coefficient
  matrix precomputed in `__init__` (`self.coef`, shape `(n_sv, n_pairs)`) and a single `torch.mm`, instead of a
  Python double loop over class pairs and support vectors. Shared by the sklearn SVC/NuSVC and ONNX-ML SVM converters.
  Test added: `tests/test_sklearn_sv_converter.py::TestSklearnSVC::test_svc_many_classes_unbalanced`.
- How to tell it is merged: `main` contains `torch.mm(k, self.coef)` in `_sv_implementations.py`.
- Pre-merge SHA (use as `--before`): f538f0769cc38334e7f676832ce8fedbae1935c3
- Re-measure: `python .github/efficiency-benchmarks/remeasure.py --before f538f0769cc38334e7f676832ce8fedbae1935c3 --after main --rounds 5`
- Pre-submission numbers (Apple M4, py3.9, torch 2.4.1, sklearn 1.2.1, 1 torch thread, 5 rounds, median predict()):
  | Config | SVs | Before ms | After ms | Speedup | CPU reduction |
  |---|---:|---:|---:|---:|---:|
  | 4cls-rbf-32f-b256 | 747 | 7.54 | 0.42 | 18.1x | 95% |
  | 8cls-linear-50f-b2000 | 1575 | 65.92 | 2.02 | 32.6x | 97% |
  | 8cls-rbf-50f-b2000 | 1838 | 80.04 | 7.74 | 10.3x | 91% |
  Predictions identical to sklearn before and after. Peak RSS unchanged (<1 MB). Expect different absolute numbers
  on Actions runners; compare ratios.
