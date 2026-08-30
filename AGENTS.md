# Hermes Realtime Development Guide

## Purpose

This repository is an independently authored realtime conversation runtime for upstream Hermes Agent. It must not copy implementation code from third-party Hermes voice projects.

## Engineering rules

- Follow strict RED-GREEN-REFACTOR for production behavior.
- Keep the protocol small, versioned, strict, and fail-closed.
- Use official Hermes and LiveKit documentation as implementation authorities.
- Keep realtime conversation separate from durable/background work.
- Preserve independent cancellation scopes for speech, foreground turns, named tasks, and sessions.
- Never claim work started before a real accepted dispatch acknowledgment.
- Never commit credentials, tokens, private URLs, transcripts, or user audio.
- Support Windows development and Linux CI/deployment.
- Prefer deterministic state machines and policies over model-controlled lifecycle state.

## Commands

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run mypy src
```

## Commit style

Use small commits with `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, or `chore:` prefixes.
