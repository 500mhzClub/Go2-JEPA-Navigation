"""Read complete, freshly written metric rows without crediting retained history."""
import json

def observed_step(path, started_at, previous):
    if not path.exists() or path.stat().st_mtime <= started_at:
        return previous
    text = path.read_text()
    complete = text.split('\n')[:-1]
    steps = [json.loads(line).get('step', 0) for line in complete if line]
    return max([previous] + steps)
