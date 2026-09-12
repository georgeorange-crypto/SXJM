from pathlib import Path
import re


def test_production_status_writes_are_confined_to_channel_belief():
    root = Path(__file__).parents[1] / "src" / "way4"
    writers = []
    for path in root.rglob("*.py"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.status\s*=(?!=)", line):
                writers.append((path.name, line_no, line.strip()))
    assert writers
    assert all(name == "channel.py" for name, _, _ in writers)
