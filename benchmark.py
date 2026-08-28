#!/usr/bin/env python3
"""
psi-agent-benchmark: Unified benchmark CLI
Supports TB 2.1, TB 3.0, tau2-bench, and GAIA as four separate benchmarks.
Each benchmark generates its own report with version-specific data format.

Usage:
  # Run a single benchmark
  python benchmark.py run -b tb-2.1
  python benchmark.py run -b tb-3.0
  python benchmark.py run -b tau2 --subset balanced_50
  python benchmark.py run -b gaia --subset level1_smoke

  # Run all four benchmarks sequentially
  python benchmark.py run -b all --run-id unified-001

  # Generate reports (each benchmark separate)
  python benchmark.py report -b tb-2.1 --run-id tb21-001
  python benchmark.py report -b tb-3.0 --run-id tb30-001
  python benchmark.py report -b all --run-id unified-001

  # List available cases/subsets
  python benchmark.py list -b tb-2.1
  python benchmark.py list -b all

  # Print data schema for a benchmark
  python benchmark.py schema -b tb-3.0
  python benchmark.py schema -b all
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from orchestrator import main

if __name__ == "__main__":
    sys.exit(main())
