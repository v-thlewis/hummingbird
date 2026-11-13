# Validated Commands (hummingbird-ml)

## 2026-11-13 - re-validated on `main` after re-fork (supersedes older notes below)
- Older notes below were collected on a stale non-default branch and are WRONG for `main`
  (e.g. `main` setup.py requires `torch>1.7.0`, not `<=1.7.0`). Ignore them unless re-verified.
- CI (`.github/workflows/pythonapp.yml`) uses Python 3.9-3.11, torch 2.4.1, numpy 1.26.4,
  scikit-learn 1.2.1 (py3.9 ubuntu) or `<1.5.0`. CI is disabled in this fork.
- Working env (validated locally on Python 3.9):
  `pip install torch==2.4.1 numpy==1.26.4 scikit-learn==1.2.1 scipy onnxconverter-common "onnx<=1.16.1" psutil dill "protobuf>=3.20.2" "onnxruntime<1.18.0" "onnxmltools<=1.12.0" "skl2onnx<=1.16.0" pytest flake8`
- Tests from repo root: `python -m pytest tests -q` (full suite ~100 s: 459 passed, 206 skipped;
  `tests/test_backends.py::TestBackends::test_onnx` fails on unmodified `main` too - pre-existing).
- Lint exactly like CI:
  `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  `flake8 . --count --max-complexity=10 --max-line-length=127 --statistics`
- Format: black, line length 127 (pyproject.toml). CI does not run black. Recent black versions reformat
  untouched lines (`x ** 2` -> `x**2`, blank line after module docstring); keep those out of PRs.
- Benchmark harness (fork only): `.github/efficiency-benchmarks/remeasure.py --before <ref> --after <ref>`
  compares two refs in temporary git worktrees, alternating rounds.

## Older notes (stale branch, pre-2026-11-13)

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
