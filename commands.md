# Validated Commands (hummingbird-ml)

- Env: repo requires torch<=1.7.0/numpy<=1.19.2/sklearn<=0.23.2 per setup.py, but those old
  wheels aren't installable on this sandbox's Python/OS. Use a fresh venv with unpinned
  `numpy scikit-learn torch onnxconverter-common psutil pytest` instead (works on Python 3.12).
- Do NOT `pip install -e .` — setup.py has an invalid version specifier
  (`torch>=1.4.*,<=1.7.0`) that fails on modern setuptools. Instead run tests with
  `PYTHONPATH=. python -m pytest tests/<file> -q -k "not tvm"` from repo root.
- Test suite: many tests fail out-of-the-box on modern sklearn/numpy (API renames like
  `sigma_`->`var_`, `n_input_features_` removed, `SGDClassifier loss="log"` removed, etc.)
  — these are PRE-EXISTING failures unrelated to any efficiency change. `tests/test_tmp_failcase.py`
  fails to even collect (ignore it: `--ignore=tests/test_tmp_failcase.py`).
- Lint: `flake8 <file> --max-complexity=10 --max-line-length=127` (repo config in .flake8).
- Format: `black <file> --line-length 127` — CAUTION: this sandbox's black version reformats
  `x ** 2` -> `x**2` repo-wide unrelated to any diff; revert such incidental hunks to keep PRs minimal.
- No benchmark runner script found beyond `benchmarks/` dir (manual scripts, not pytest-integrated).

## Update 2026-09-03
- Sandbox default `python3` is PyPy 3.9 (no scipy/sklearn wheels available, build fails on
  OpenBLAS/meson). Use CPython instead: `/opt/hostedtoolcache/Python/3.12.14/x64/bin/python3 -m venv <path>`
  then `pip install numpy scikit-learn torch onnxconverter-common psutil pytest packaging flake8 black dill`
  (dill needed for tvm_containers import chain even when tvm itself isn't installed).
