from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rererank_v1.benchmark_data import DOCUMENTS, TEST_CASES, TestCase

__all__ = ["TestCase", "DOCUMENTS", "TEST_CASES"]

