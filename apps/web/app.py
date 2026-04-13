from flask import Flask, render_template, jsonify, send_from_directory
import json
import os
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_FILE = DATA_DIR / "research_history.json"
ABLATION_FILE = DATA_DIR / "results" / "automated_ablation.json"
IMAGE_DIR = Path(__file__).resolve().parents[2] / "docs" / "images"

def load_data():
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"iterations": [], "plan": []}

def load_ablation_data():
    if ABLATION_FILE.exists():
        with open(ABLATION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

@app.route('/')
def index():
    data = load_data()
    ablation = load_ablation_data()
    return render_template('dashboard.html', data=data, ablation=ablation)

@app.route('/api/data')
def get_data():
    return jsonify(load_data())

@app.route('/api/ablation')
def get_ablation_data():
    return jsonify(load_ablation_data())

@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    # Add initial plan if empty
    data = load_data()
    if not data.get("plan"):
        data["plan"] = [
            {"phase": "Phase 1", "task": "Baseline Vector Retrieval", "status": "Completed", "date": "2024-03-14"},
            {"phase": "Phase 2", "task": "Integrate BGE Reranker", "status": "Completed", "date": "2024-03-14"},
            {"phase": "Phase 3", "task": "Implement PRF & RRF", "status": "Completed", "date": "2024-03-14"},
            {"phase": "Phase 4", "task": "Robustness Testing (Ambiguity)", "status": "In Progress", "date": "2024-03-14"},
            {"phase": "Phase 5", "task": "Real-world Deployment", "status": "Planned", "date": "TBD"}
        ]
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    app.run(debug=True, port=5002)
