#!/usr/bin/env bash
# Reproduce every result that is backed by validated, deterministic code.
# Usage: scripts/reproduce_all.sh            (from anywhere; ~1 minute)
# Outputs: outputs/repro/  and outputs/constraint_tests/closure_stress_test.json
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0

echo "== environment =="
python3 -c "import sys, numpy, scipy, sympy, torch; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| sympy', sympy.__version__, '| torch', torch.__version__)"
git rev-parse --short HEAD 2>/dev/null | sed 's/^/commit /' || true

echo "== 1. regression tests (checker validation, AMR known-answer tests) =="
python3 -m pytest tests/test_closure_stress.py tests/test_amr.py -q

echo "== 2. constraint closure stress scan =="
python3 scripts/closure_stress_test.py --n 128 --states 4 --out outputs/constraint_tests/closure_stress_test.json | tail -3

echo "== 3. AMR convergence study =="
python3 scripts/amr_convergence_study.py

echo "== 4. holonomy prescription comparison (Schwarzschild) =="
mkdir -p outputs/repro/prescriptions
( cd outputs/repro/prescriptions && python3 -I ../../../alternative_polymer_prescriptions.py --schwarzschild --output schwarzschild.csv \
    | grep -v "execution time\|completed in" > schwarzschild_log.txt )
echo "   log: outputs/repro/prescriptions/schwarzschild_log.txt"

echo "done."
