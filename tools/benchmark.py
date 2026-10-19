"""Atalho para o benchmark: equivale a `pest-node benchmark ...`.

Exemplos (no Pi, dentro do venv):
    python tools/benchmark.py --duration 600
    python tools/benchmark.py --duration 600 --threads 4
    python tools/benchmark.py --duration 600 --adaptive
    python tools/benchmark.py --duration 60 --images ../servidor_cv/data   # câmera falsa
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pest_node.bench import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
