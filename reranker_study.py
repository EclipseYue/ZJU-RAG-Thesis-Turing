import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "experiments" / "reranker_study.py"), run_name="__main__")

