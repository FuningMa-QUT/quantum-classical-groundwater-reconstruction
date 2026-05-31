"""Run a quantum_hydro experiment config from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_hydro.experiments import main


if __name__ == "__main__":
    main()

