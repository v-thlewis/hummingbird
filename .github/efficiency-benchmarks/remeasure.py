"""Compare a benchmark between two git refs (e.g. main before and after an upstream merge).

Each ref is checked out into a temporary git worktree and benchmarked in a fresh subprocess with
that worktree first on PYTHONPATH. Rounds alternate before/after to spread out machine noise.

Example:
    python .github/efficiency-benchmarks/remeasure.py --before <pre-merge-sha> --after upstream/main
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "svc_predict_bench.py")
METRICS = ["wall_ms_median", "cpu_ms_median", "peak_rss_mb"]


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def bench(worktree, bench_args):
    env = dict(os.environ, PYTHONPATH=worktree)
    out = subprocess.run(
        [sys.executable, BENCH, *bench_args], cwd=tempfile.gettempdir(), env=env, check=True, capture_output=True, text=True
    ).stdout
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    for row in rows:
        if not row["hummingbird_path"].startswith(os.path.realpath(worktree)):
            raise RuntimeError("Imported hummingbird from {} instead of {}".format(row["hummingbird_path"], worktree))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json-out")
    args, bench_args = parser.parse_known_args()

    repo = git("rev-parse", "--show-toplevel", cwd=HERE)
    refs = {"before": git("rev-parse", args.before, cwd=repo), "after": git("rev-parse", args.after, cwd=repo)}
    tmp = tempfile.mkdtemp(prefix="hb-remeasure-")
    trees = {}
    try:
        for label, sha in refs.items():
            trees[label] = os.path.realpath(os.path.join(tmp, label))
            git("worktree", "add", "--detach", trees[label], sha, cwd=repo)

        results = {"before": [], "after": []}
        for i in range(args.rounds):
            for label in (["before", "after"] if i % 2 == 0 else ["after", "before"]):
                results[label].extend(bench(trees[label], bench_args))
                print("round {} {} done".format(i + 1, label), file=sys.stderr, flush=True)
    finally:
        for path in trees.values():
            subprocess.run(["git", "worktree", "remove", "--force", path], cwd=repo, capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)

    summary = {"refs": refs, "rounds": args.rounds, "configs": {}}
    for config in sorted({r["config"] for r in results["before"]}):
        entry = {}
        for label in ("before", "after"):
            rows = [r for r in results[label] if r["config"] == config]
            entry[label] = {m: statistics.median(r[m] for r in rows) for m in METRICS}
            entry[label]["predictions_match_sklearn"] = all(r["predictions_match_sklearn"] for r in rows)
            entry[label]["n_support_vectors"] = rows[0]["n_support_vectors"]
        entry["wall_speedup_x"] = entry["before"]["wall_ms_median"] / entry["after"]["wall_ms_median"]
        entry["cpu_reduction_pct"] = 100 * (1 - entry["after"]["cpu_ms_median"] / entry["before"]["cpu_ms_median"])
        summary["configs"][config] = entry

    print("| Config | SVs | Before wall ms | After wall ms | Speedup | CPU reduction | Before RSS MB | After RSS MB | Predictions match |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for config, e in summary["configs"].items():
        b, a = e["before"], e["after"]
        print(
            "| {} | {} | {:.2f} | {:.2f} | {:.1f}x | {:.0f}% | {:.0f} | {:.0f} | {} |".format(
                config,
                b["n_support_vectors"],
                b["wall_ms_median"],
                a["wall_ms_median"],
                e["wall_speedup_x"],
                e["cpu_reduction_pct"],
                b["peak_rss_mb"],
                a["peak_rss_mb"],
                b["predictions_match_sklearn"] and a["predictions_match_sklearn"],
            )
        )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
