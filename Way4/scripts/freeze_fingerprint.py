"""Create a deterministic source-tree fingerprint for a non-clean Git snapshot."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (ROOT / "src", ROOT / "tests", ROOT / "DESIGN.md", ROOT / "FREEZE_MANIFEST.md")
OUT = ROOT / "results" / "way4_source_fingerprint.json"

def files():
    out=[]
    for item in INCLUDE:
        if item.is_file(): out.append(item)
        elif item.is_dir(): out.extend(p for p in item.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sorted(out)

def main():
    rows=[]
    for p in files():
        rows.append({"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "bytes": p.stat().st_size})
    tree = hashlib.sha256("".join(r["path"] + r["sha256"] for r in rows).encode()).hexdigest()
    payload={"kind":"way4-source-tree-fingerprint-v1","git_head":"399d6ad",
             "working_tree_has_uncommitted_changes":True,"file_count":len(rows),
             "tree_sha256":tree,"files":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"out":str(OUT),"file_count":len(rows),"tree_sha256":tree}))
if __name__ == "__main__": main()
