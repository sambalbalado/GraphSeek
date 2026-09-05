# Development

```sh
pytest
ruff check .
mypy src
```

Keep changes focused. Add a test demonstrating the behavior, read the diff,
and commit after checks pass. Record progress when it happens.

If using task branches, merge each verified branch into main and push main
before creating the next branch. Keep datasets and credentials out of Git.
