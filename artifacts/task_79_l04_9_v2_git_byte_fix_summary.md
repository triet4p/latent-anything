# Task Summary: Sprint 79 L04.9 v2 Git byte-integrity fix

**Sprint:** Sprint 79
**Task:** L04.9 v2 cross-clone evidence byte integrity

## Summary

Scoped `-text` attributes now cover the L04.9 v2 canonical evidence family,
including the official D3 record. Canonical promotion reads and tracked Stage B
preflight fail closed unless Git reports `text: unset`, preventing
`core.autocrlf=true` from changing pinned evidence bytes. Future Stage B
build-only/fake behavior is unchanged.

## Files

- `.gitattributes` — scoped L04.9 v2 binary attributes.
- `scripts/_m14_l049_v2_inputs.py` — Stage B preflight attribute guard.
- `scripts/_m14_l049_v2_promotion.py` — canonical and bound-file read guards.
- `tests/test_m14_l049_v2_git_integrity.py` — attr/blob/clone regressions.
- `tests/test_m14_l049_v2.py` — temporary Git fixture carries the attr contract.
- `docs/M14_REAL_SYSTEM_VALIDATION.md`, `CHANGELOG.md`, and memory records —
  contract and lesson documentation.

## Verification

- Git attr contract: 18 canonical real-promotion files plus D3 report
  `text: unset`.
- Would-be Git blob checks: every tracked canonical path's `--path` and
  `--no-filters` OIDs equal the index OID; D3 raw/path OIDs equal.
- Local temporary clone with `core.autocrlf=true`: all canonical SHA-256 values
  and sizes unchanged; residue cleaned by pytest `tmp_path`.
- `uv run pytest tests/test_m14_l049_v2_git_integrity.py tests/test_m14_l049_v2_validation_context.py -q` — 11 passed.
- `uv run pytest tests/test_m14_l049_v2.py -q` — 212 passed after fixture
  contract update.

The official D3 artifact remains unchanged at 4032 bytes with SHA-256
`a9444cf7afc720e5db5961227cff275e24f7cd80bfcedfdbafc74aa1874de6b6`.

## Immutable hash snapshot

The before/after snapshots are identical:

| Evidence | Bytes | SHA-256 |
| --- | ---: | --- |
| D1 assessment | 7447 | `735d7fca2a157aaaefdcbb2667b95ff9fd91f6445b70c78cc9cee1e82b790d66` |
| D1 audit | 3397 | `a1b60ec6804e0468716398c75c9e3508a1c982c0b312fcd8fb1c5aab737e166d` |
| D1 candidate | 526043 | `29bcd20ab494092abbb074bff5d99d091ec288d261a0399f97f2e2fb4f092aa2` |
| D2 assessment | 9663 | `1fc621818f89c932dc46d0f80ca22aa2aaabf1f19c869a62fb0bcf71b818070f` |
| D2 audit | 3395 | `c8a308655103a75845ae45a0cc0a8029408958e4c9c01335db1a22854b0cef85` |
| Provisioning assessment | 3297 | `7bbc7276a44cc2ae0e68a2e6ca09c35d22f718c94392c7e712bc3ac0f9a0804c` |
| Manifest | 656 | `2849b07fd719a0a761f433892fcc031c2ab17012a538daba322dd6fa50674974` |
| Holdout | 18312 | `295ef5f558315c629d68e2d0216567a67163e5ef4adaaf3bbc9fe8a4da96dd5f` |
| Holdout seed | 32 | `b8e5e28908c2d2925a5bf5dcc69d852b4e31584f23f0ced2903a70f10d36b5e1` |
| Train | 23076 | `f4cb7b52f946263a99113b9ebd8b24a74f66b49cd17fce77c15a044ec671a9e9` |
| D1 failure | 399 | `a40f645d7e8cbb6ccf76765287ff09d592b70ea9f5e4d284e1a5c9c74d489afe` |
| D1 partial | 526385 | `f5fff08f0de818bb4ef91157b7e94d9c200343afcc3e6c53444b041fa840eee2` |
| D1 run | 397 | `0123b4dbd38b921c5174dfcb87c2e5bd08fdd08cc66db4374191d31a061fed9a` |
| D2 failure | 402 | `b8c2000afbec9900f706034dfe742761ef95034f94aa9606b49c9686336144e2` |
| D2 partial | 113808 | `18f60e97ce21ff88a1fa27c1b3e23e0f3bbcc7898ee7440e1a1804b4f695f0eb` |
| D2 run | 400 | `929c53129bf8285055a689c11da05675f473cd4a5984bc8efe82a8f34886a210` |
| Official D3 | 4032 | `a9444cf7afc720e5db5961227cff275e24f7cd80bfcedfdbafc74aa1874de6b6` |
