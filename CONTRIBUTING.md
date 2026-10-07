# Contributing to Omni Slated Mem Core

Thanks for helping build a deterministic local memory system.

## Ground rules

- **Deterministic.** Identical inputs produce byte-identical results, or it
  does not ship. The smoke test asserts this.
- **HTTP 200 only when searchable.** `POST /v1/memories/add` returns 200 only
  after the messages are stored **and** searchable (synchronous durability).
- **Zero LLM.** No language-model calls at any stage — no API keys, no
  per-query inference cost. Keep it that way.
- **`user_id` is the isolation boundary.** A user only ever retrieves their
  own memories. Never weaken this.

## Quick check (the whole gate, one command)

```sh
python scripts/smoke_test.py --port 18001   # boots its own server
```

CI runs this, a version-agreement check, and a secret scan on every pull
request.

## Changing retrieval behavior

1. Re-run `scripts/calibrate.py` (threshold) and
   `scripts/calibrate_contra.py` (contradiction precision/recall).
2. Update the Calibration section of `README.md` if the reported numbers
   move.
3. The smoke test must still pass all 25 checks, including determinism.

## Changing the API

- New endpoints must not weaken the contract: add-only, searchable-before-200,
  user-scoped. Document them in `README.md` and in
  `GET /service/about/endpoints`.
- Open a pull request using the template.

## Versioning

This repo tags releases as `vX.Y.Z`. When cutting a release, bump these
together: the README title, `app/config.py` `SYSTEM_VERSION`, the
`app/main.py` docstring, and the Dockerfile `LABEL
org.opencontainers.image.version`. CI verifies they agree.
