from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from apps.web.app import app

if __name__ == "__main__":
    app.run(debug=True, port=5001)
