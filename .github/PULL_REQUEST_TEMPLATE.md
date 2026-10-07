## What changed

<!-- One or two sentences. -->

## Checks

- [ ] `python scripts/smoke_test.py --port 18001` passes (boots its own server)
- [ ] `scripts/calibrate.py` and `scripts/calibrate_contra.py` still report the documented thresholds, if retrieval behavior changed
- [ ] Determinism holds: identical inputs produce byte-identical results
- [ ] `user_id` isolation intact: no user can retrieve another user's memories
- [ ] Version strings updated together (README title, `app/config.py` `SYSTEM_VERSION`, `app/main.py` docstring, Dockerfile `LABEL`) if this is a release
