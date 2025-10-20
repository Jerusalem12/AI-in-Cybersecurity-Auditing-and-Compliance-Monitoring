import json
from pathlib import Path
from datetime import datetime

def read_json(path: str | Path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}

def write_json(obj, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)
