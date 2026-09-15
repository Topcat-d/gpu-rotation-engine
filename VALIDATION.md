# Validation scope

This standalone native SDK supports continued development independently of Smoke
Suite. The existing `smoke_*` C ABI names remain for compatibility.

The validated profile is sequential AES-256-GCM, ChaCha20-Poly1305 and P-256 on
Windows, CUDA 12.4, Visual Studio 2022, and an RTX 4070 Ti (`sm_89`), with
`SMOKE_FASTPATH_FULLWINDOW=ON`. Other included primitives are not covered by this
release's end-to-end evidence. Linux build instructions are provided but have not
been validated in this release.

The table tests regenerate both comb widths and verify all 4,224 full-window
points independently using pyca/cryptography. The installed example checks
encryption/decryption against an independent CPU implementation, key rotation,
rollback, asynchronous fences, invalid ciphertexts, and old-key restoration.

This is a technical preview. These checks do not establish concurrency safety,
side-channel resistance, throughput, or production certification. The native
profile has no caller-supplied AAD. Applications own nonce uniqueness, key custody,
permanent key IDs, and historical key retention. Rotation does not automatically
re-encrypt previously stored data.

Only source files and public precomputation tables are distributed. Build locally;
no historical compiled libraries or private repository history are included.

## Verified standalone snapshot — 2026-09-15

Fresh native configuration, build, and installation passed. All 3 table tests
and both native CTests passed. The installed example ran from outside this
checkout using only its declared Python dependencies: 616 GPU operations, 120
AEAD round trips, 12 P-256 signatures, and 24 archived-record restorations.
See [the machine-readable report](evidence/standalone-20260915.json).

