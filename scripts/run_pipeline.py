"""
Run the full photometry analysis pipeline.

Executes each stage in order:
1. query_database  — fetch session metadata from Alyx → sessions.pqt
2. photometry      — QC + preprocess + responses → HDF5 files + qc_photometry.pqt
3. task            — compute task performance → performance.pqt
4. wheel           — extract per-trial wheel velocity → HDF5 files
5. dataset_overview — aggregate errors + generate figures → errors.pqt

Each stage is run as a subprocess so that argparse, top-level code, and
matplotlib state don't leak between scripts.

Usage:
    python scripts/run_pipeline.py                  # run all stages
    python scripts/run_pipeline.py --from photometry  # resume from photometry
    python scripts/run_pipeline.py --only task        # run single stage
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STAGES = [
    'query_database',
    'photometry',
    'task',
    'wheel',
    'dataset_overview',
]

STAGE_SCRIPTS = {stage: SCRIPTS_DIR / f'{stage}.py' for stage in STAGES}


def run_stage(stage):
    """Run a pipeline stage as a subprocess. Raises on failure."""
    script = STAGE_SCRIPTS[stage]
    if not script.exists():
        print(f"  SKIP: {script} not found")
        return False

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SCRIPTS_DIR.parent),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s, exit code {result.returncode})")
        return False

    print(f"  done ({elapsed:.0f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run photometry analysis pipeline')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--from', dest='from_stage', choices=STAGES, metavar='STAGE',
                       help='Resume from this stage (inclusive)')
    group.add_argument('--only', choices=STAGES, metavar='STAGE',
                       help='Run only this stage')
    parser.add_argument('--skip-errors', action='store_true',
                        help='Continue to next stage even if one fails')
    args = parser.parse_args()

    if args.only:
        stages = [args.only]
    elif args.from_stage:
        start = STAGES.index(args.from_stage)
        stages = STAGES[start:]
    else:
        stages = STAGES

    print(f"Pipeline: {' → '.join(stages)}\n")

    failed = []
    for stage in stages:
        print(f"[{stage}]")
        ok = run_stage(stage)
        if not ok:
            failed.append(stage)
            if not args.skip_errors:
                print(f"\nPipeline stopped at {stage}. Use --skip-errors to continue past failures.")
                sys.exit(1)
        print()

    if failed:
        print(f"Completed with failures: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("Pipeline complete.")


if __name__ == '__main__':
    main()
