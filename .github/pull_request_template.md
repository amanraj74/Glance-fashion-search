---
name: Pull request
about: Propose a change to the codebase
title: ""
---

## What this PR does

A short, plain-English summary.

## Why

What problem does this solve? Link the issue if there is one.

## How to verify

Steps a reviewer can follow to confirm the change works:

1. Run: `python ...`
2. With: `...`
3. Expect: `...`

## Checklist

- [ ] `pytest tests/ -v` passes locally
- [ ] New code is covered by tests where applicable
- [ ] `README.md` and `CHANGELOG.md` are updated if user-facing behaviour changes
- [ ] No secrets, tokens, or large model weights are committed
- [ ] Commits are squashed and the title is imperative ("Add X", not "Added X")
