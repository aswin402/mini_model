# MIVI Foundation Evaluation

This evaluation checks the first cognitive-kernel foundation without opening or modifying `data/little.db`.

It measures:

- registry-backed action validation;
- UNKNOWN relation non-commit behavior;
- provenance on accepted relations;
- deterministic operation on a temporary SQLite database.

Run it with:

```bash
.venv/bin/python experiments/006_mivi_foundation/run.py
```
