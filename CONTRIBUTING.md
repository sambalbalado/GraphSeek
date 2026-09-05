# Development Workflow

## One task, one commit

1. Select the next unfinished task in `TASKS.md`.
2. Create a branch named `task/NN-short-description`.
3. Run the existing checks before editing.
4. Implement only that task and its tests.
5. Review the diff and update relevant documentation.
6. Run `pytest`, `ruff check .`, and `mypy src`.
7. Commit using the message given in the task.

Do not manufacture commits by splitting already-finished code after the fact.
The history should reflect genuine, reviewable progress.

## Working with an AI coding assistant

- Paste only one task prompt at a time.
- Ask the assistant to inspect existing code before proposing changes.
- Require tests and an explanation of time and space complexity.
- Do not accept a change you cannot explain.
- Review the diff before allowing a commit.
- Never commit secrets, downloaded datasets, environments, or benchmark output.

