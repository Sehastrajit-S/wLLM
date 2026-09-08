"""Runs the test suite in fixed-size batches, each its own pytest subprocess.

Why this exists: the suite loads dozens of real models (some GB each, more
with CUDA graph capture) across 30+ files sharing a single pytest process.
tests/conftest.py's autouse empty_cache() fixture keeps any INDIVIDUAL file
bounded, but several files use module-scoped fixtures that hold a full model
resident for that whole file's run -- chain enough heavy files together in
one process on a 12GB card and it still runs out, independent of which
kernel/model code is under test (reproduces with an unmodified kernel too).
Splitting into separate subprocesses is the actual fix: each batch's peak
usage can't carry over into the next, since the OS reclaims everything when
a process exits.

Run: python scripts/run_tests.py [pytest args, e.g. -k foo]
Batch size is tuned empirically (8 files/batch was reliable during
development on a 12GB RTX 3060 -- lower it if a batch still OOMs on smaller
hardware, raise it on a bigger card for faster runs).
"""
import subprocess
import sys

BATCH_SIZE = 8


def main() -> int:
    extra_args = sys.argv[1:]

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/"],
        capture_output=True, text=True,
    )
    test_files = sorted({line.split("::")[0] for line in result.stdout.splitlines() if "::" in line})
    if not test_files:
        print("no test files discovered -- collection may have failed:\n" + result.stdout + result.stderr)
        return 1

    batches = [test_files[i : i + BATCH_SIZE] for i in range(0, len(test_files), BATCH_SIZE)]
    print(f"running {len(test_files)} files in {len(batches)} batches of up to {BATCH_SIZE}")

    overall_ok = True
    for i, batch in enumerate(batches):
        print(f"\n=== batch {i + 1}/{len(batches)}: {', '.join(batch)} ===")
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", *extra_args, *batch])
        overall_ok = overall_ok and proc.returncode == 0

    print("\n=== ALL BATCHES PASSED ===" if overall_ok else "\n=== ONE OR MORE BATCHES FAILED ===")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
