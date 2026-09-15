# GPU Rotation Engine

This repository includes the GPU engine, C ABI, encryption/decryption kernels,
rotation implementation, precomputation tables, generators, and runnable tests.
The standalone SDK does not require Smoke Suite, its gateway, or PyTorch.

**Technical preview.** The walkthrough validates AES-256-GCM, ChaCha20-Poly1305,
and P-256 through the installed native library. It does not validate every
historical experiment or optional primitive.

## Build and install

Requirements: NVIDIA GPU, CUDA toolkit, CMake 3.18+, and a compatible C/C++
toolchain. Windows also needs Visual Studio C++ build tools and the Windows SDK.

```sh
git clone https://github.com/Topcat-d/gpu-rotation-engine.git
cd gpu-rotation-engine
cmake -S . -B build_standalone -DCMAKE_BUILD_TYPE=Release -DSMOKE_FASTPATH_FULLWINDOW=ON -DSMOKE_CUDA_ARCHITECTURES=89
cmake --build build_standalone --config Release --parallel 4
cmake --install build_standalone --config Release --prefix install
```

On Windows, add `-G "Visual Studio 17 2022" -A x64` to configuration if needed.
Architecture `89` targets the RTX 4070 Ti used for validation. Select your GPU's
architecture elsewhere; omit that option to build `70;75;80;86;89;90`.
Only `sm_89` with the full-window signer enabled was validated for this release.
The full-window option is saved across subsequent builds.

The installed SDK contains:

| Path under `install/` | Contents |
|---|---|
| `bin/` or `lib/` | Native engine and platform link artifacts |
| `include/` | C ABI and rotation-probe headers |
| `share/smoke_engine/data/p256_comb/` | Width-7 and width-8 X/Y comb tables |
| `share/smoke_engine/data/p256_fullwindow/` | Full-window table and checksum |
| `share/smoke_engine/tools/` | Reproducible table generators |
| `share/smoke_engine/examples/` | End-to-end validation program |
| `share/smoke_engine/requirements/` | Validation dependency list |
| `share/licenses/smoke_engine/` | Apache-2.0 license and MIT notices |

The tables contain public generator-point multiples for P-256 signing, not secret
keys. AES/ChaCha do not depend on them. The current P-256 loader uses width 8.
Full-window loading checks the typed header, compiled SHA-256 fingerprint, and
device readback before accepting the table.

## Run the complete installed-SDK check

The Python program calls the C ABI using `ctypes`. `cryptography` is the
independent test oracle; it is not a CPU fallback for the GPU engine.

```sh
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r install/share/smoke_engine/requirements/standalone.txt
```

Windows:

```powershell
python install/share/smoke_engine/examples/standalone_rotation.py --library install/bin/smoke_engine_abi.dll --data-dir install/share/smoke_engine/data --require-fullwindow --report rotation-report.json
```

Linux library path (not validated by this Windows release run):

```sh
python install/share/smoke_engine/examples/standalone_rotation.py --library install/lib/libsmoke_engine_abi.so --data-dir install/share/smoke_engine/data --require-fullwindow --report rotation-report.json
```

The default check performs:

- Table uploads, refusal of unsafe upload order, and corrupted-table rejection.
- 12 P-256 signatures checked against the intended key across two rotations.
- 120 AES-GCM/ChaCha20-Poly1305 round trips across four rotations: two synchronous
  and two asynchronous, with prepare/commit/fence and rollback checks.
- Empty, inline, and slab messages through 65,535 plaintext bytes, with ciphertext
  and tags compared against the independent oracle.
- Wrong-key and tampered-ciphertext rejection with no returned plaintext.
- Previous-key decryption and restoration of 24 archived records after live slots
  are reused, without damaging the active key.

It exits nonzero on failure and has a 120-second watchdog for hung native calls.
Keys are randomly generated in memory and not written to the report. This is a
sequential correctness test, not a throughput or concurrency test.

## Correct native call order

1. `smoke_engine_init` creates the engine and starts its legacy kernel.
2. `smoke_fast_path_create` stops that kernel and allocates fast-path resources.
3. Upload the comb and, when enabled, full-window tables. Uploading while either
   persistent kernel runs is rejected.
4. Load keys, configure matching service lanes and shards (the example uses two
   each), and call `smoke_fast_path_start`.
5. Submit encryption/decryption with an explicit `key_slot`. Copy each response
   before reusing its payload/output space.
6. Prepare the next typed key, commit, and fence after asynchronous calls before
   submitting more work. Read the active slot with `smoke_engine_rotation_probe`.
7. Stop and shut down when finished.

The example serializes submission and slot reuse. Concurrent applications must
coordinate in-flight requests before overwriting a keyslot.

## Key lifetime and ciphertext storage

Rotation changes future operations' keys. It does **not** automatically re-encrypt
stored data or archive old keys. The rotation pair has two reusable slots; a slot
index is not a permanent key ID.

Store ciphertext with its algorithm, immutable key ID, nonce, and tag. Keep older
keys in your own secure key manager while their ciphertext is needed, or decrypt
and re-encrypt those records before retiring the key. The example's in-memory
archive demonstrates restoring a key into the inactive slot. It is not durable
storage and disappears when the program exits.

Use a unique nonce for each encryption under a given key. This native profile
does not accept caller-supplied additional authenticated data (AAD); the test uses
none. Authenticating application metadata needs an application format and policy
beyond the raw ciphertext demonstrated here.

## Tests and continued optimization

```sh
python -m pip install pytest -r requirements/standalone.txt
python -m pytest -q tests/test_standalone_tables.py
ctest --test-dir build_standalone -C Release --output-on-failure --timeout 45
```

On Windows, put `build_standalone/bin/Release` and CUDA's `bin` directory on `PATH`
for CTest. The native smoke test checks a known SHA-256 digest. CPU CI regenerates
both comb widths and verifies all 4,224 full-window points against pyca.
