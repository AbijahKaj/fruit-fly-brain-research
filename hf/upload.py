"""Push the graph, the parameter files, the training logs and the model card to the Hub.

    python hf/upload.py            # uploads to AbijahKaj/fruit-fly-brain
    python hf/upload.py org/name   # another repo id

Needs `huggingface_hub` and a login (`hf auth login`).
"""
from __future__ import annotations
import sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

ROOT = Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "app" / "public" / "graphs"
repo_id = sys.argv[1] if len(sys.argv) > 1 else "AbijahKaj/fruit-fly-brain"

files = {
    "README.md": ROOT / "hf" / "README.md",
    "optic.json": GRAPHS / "optic-v2.json",
    "optic.bin": GRAPHS / "optic-v2.bin",
    "fitted-params.json": GRAPHS / "fitted-params.json",
    "flyvis-params.json": GRAPHS / "flyvis-params.json",
}
for log in sorted((ROOT / "train" / "runs").glob("*.log")):
    files[f"runs/{log.name}"] = log

api = HfApi()
api.create_repo(repo_id, repo_type="model", exist_ok=True)
ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
api.create_commit(repo_id=repo_id, operations=ops, commit_message="Graph, fitted parameters, training logs, model card")
print(f"https://huggingface.co/{repo_id}")
