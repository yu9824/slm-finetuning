"""Root conftest.py: add project root to sys.path for src imports."""

import sys
from pathlib import Path

# Ensure `src` package is importable as `src.*`
sys.path.insert(0, str(Path(__file__).parent))
