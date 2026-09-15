#!/usr/bin/env python3
"""
P-256 Full-Window Fixed-Base Table Generator (zero-doubling scalar mul)

Generates the table for the full-window fixed-base method: for every window
position i in [0, 32) and every digit d in [1, 128], the affine point

    T[i][d-1] = (d * 2^(8*i)) * G

in the Montgomery domain (x*R mod p), little-endian u32 limbs.

With signed base-256 recoding (digits in [-128, 127], carry-propagated),
k*G = sum_i digit_i * 2^(8i) * G needs only 32 table lookups + 32 mixed
adds and ZERO doublings. Negative digits use the negated-Y of T[i][|d|-1];
|d| reaches 128, hence 128 stored multiples per window.

Outputs:
  p256_G_fullwindow_w8.bin   AoS: entry = X[8 u32 LE] || Y[8 u32 LE] (64 B),
                             index = window*128 + (digit-1); 4096 entries,
                             262,144 bytes total. For //go:embed + C ABI load.
  p256_G_fullwindow_w8.sha256  fingerprint of the .bin (table-integrity gate)
  p256_G_fullwindow_w8_meta.json  layout metadata + golden vectors

Every point is INDEPENDENTLY verified against a second implementation
(pyca/cryptography public-key derivation for the scalar d*2^(8i)), so a bug
in the local affine math cannot silently ship. This is generator-side
tooling only — no GPU code, no engine changes. (Card guardrail: two
independent generators must agree byte-for-byte.)

Usage:
    python gen_p256_fullwindow_table.py --outdir data/p256_fullwindow
"""

import os
import sys
import json
import hashlib
import argparse

import numpy as np

# Reuse the validated curve/limb/Montgomery helpers from the comb generator
# (CARD17-correct to_montgomery: x*R, not x*R^2).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_p256_comb_table import (  # noqa: E402
    P, Gx, Gy, N,
    int_to_limbs_le, to_montgomery,
    point_add_affine, point_double_affine,
)

# 33 windows, NOT 32: the signed fixed-window recode of a 256-bit scalar can
# emit a final carry digit at window 32 (weight 2^256) when W=8 divides 256
# (recode_signed_fixed_window_warp emits digits[32]=1 on final carry). The
# comb method absorbs that carry with doublings; the zero-doubling method must
# store window 32 explicitly or it produces wrong signatures on carrying
# scalars (~1/256). Window 32 only needs digit 1 in practice, but we store the
# full 128 for uniform indexing.
WINDOWS = 33          # 32 base windows + 1 carry window (2^256*G)
DIGITS = 128          # stored multiples 1..128 (|signed digit| max)
ENTRY_BYTES = 64      # X(32B) || Y(32B)

# Typed blob header (architect condition C3, SPEED-H2-01): the loader rejects
# any blob whose magic/type/dims don't match — a comb blob can never be fed
# to the fullwindow loader or vice versa. 32 bytes, all fields LE u32 after
# the magic. SHA-256 fingerprint covers header + payload (the whole file).
HEADER_MAGIC = b"SMKTBL01"          # 8 bytes
TABLE_TYPE_FULLWINDOW_W8 = 2        # 1 is reserved for the legacy comb table
HEADER_VERSION = 1


def build_header() -> bytes:
    import struct
    return HEADER_MAGIC + struct.pack(
        "<6I",
        TABLE_TYPE_FULLWINDOW_W8,
        HEADER_VERSION,
        WINDOWS,
        DIGITS,
        ENTRY_BYTES,
        0,  # reserved
    )


def scalar_base_point(i_window: int) -> tuple:
    """Return 2^(8*i) * G by repeated doubling from G (i*8 doublings)."""
    x, y = Gx, Gy
    for _ in range(8 * i_window):
        x, y = point_double_affine(x, y)
    return x, y


def gen_table() -> list:
    """Compute all WINDOWS*DIGITS affine points in the normal domain."""
    points = []
    base_x, base_y = Gx, Gy  # 2^(8*0) * G
    for i in range(WINDOWS):
        # digit 1 = base
        win = [(base_x, base_y)]
        cx, cy = base_x, base_y
        for _ in range(2, DIGITS + 1):
            cx, cy = point_add_affine(cx, cy, base_x, base_y)
            win.append((cx, cy))
        points.append(win)
        if i < WINDOWS - 1:
            # next window base = 2^8 * current base (8 doublings)
            for _ in range(8):
                base_x, base_y = point_double_affine(base_x, base_y)
        print(f"  window {i:2d}: {DIGITS} points done")
    return points


def verify_against_pyca(points: list) -> None:
    """Independent check: every point equals pyca's (d*2^(8i) mod n)*G."""
    from cryptography.hazmat.primitives.asymmetric import ec

    curve = ec.SECP256R1()
    checked = 0
    for i in range(WINDOWS):
        for d in range(1, DIGITS + 1):
            scalar = (d << (8 * i)) % N
            pub = ec.derive_private_key(scalar, curve).public_key()
            nums = pub.public_numbers()
            ex, ey = points[i][d - 1]
            if (nums.x, nums.y) != (ex, ey):
                raise SystemExit(
                    f"MISMATCH window={i} digit={d}: "
                    f"local=({ex:#x},{ey:#x}) pyca=({nums.x:#x},{nums.y:#x})"
                )
            checked += 1
        print(f"  pyca cross-check window {i:2d}: OK ({checked}/{WINDOWS*DIGITS})")
    print(f"  ALL {checked} points verified against pyca/cryptography")


def pack_aos(points: list) -> bytes:
    """Montgomery domain, AoS: per entry X limbs LE then Y limbs LE."""
    out = bytearray()
    for i in range(WINDOWS):
        for d in range(1, DIGITS + 1):
            x, y = points[i][d - 1]
            out += int_to_limbs_le(to_montgomery(x)).tobytes()
            out += int_to_limbs_le(to_montgomery(y)).tobytes()
    assert len(out) == WINDOWS * DIGITS * ENTRY_BYTES
    return bytes(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", default="data/p256_fullwindow")
    ap.add_argument("--skip-pyca", action="store_true",
                    help="skip the independent pyca verification (NOT for "
                         "release artifacts — dual verification is the gate)")
    args = ap.parse_args()

    print("=" * 72)
    print("P-256 Full-Window Fixed-Base Table Generator (32 x 128, Montgomery)")
    print("=" * 72)

    print("Computing points (local affine math)...")
    points = gen_table()

    if args.skip_pyca:
        print("WARNING: pyca cross-check SKIPPED — artifact is NOT release-grade")
    else:
        print("Cross-checking every point against pyca/cryptography...")
        verify_against_pyca(points)

    blob = build_header() + pack_aos(points)
    fp = hashlib.sha256(blob).hexdigest()

    os.makedirs(args.outdir, exist_ok=True)
    bin_path = os.path.join(args.outdir, "p256_G_fullwindow_w8.bin")
    with open(bin_path, "wb") as f:
        f.write(blob)
    with open(bin_path.replace(".bin", ".sha256"), "w") as f:
        f.write(fp + "\n")

    # Golden vectors for kernel-side canary: (window, digit, X_mont, Y_mont)
    goldens = []
    for (i, d) in [(0, 1), (0, 128), (7, 64), (15, 1), (31, 128)]:
        x, y = points[i][d - 1]
        goldens.append({
            "window": i, "digit": d,
            "x_mont": f"{to_montgomery(x):064x}",
            "y_mont": f"{to_montgomery(y):064x}",
        })
    meta = {
        "header": {"magic": HEADER_MAGIC.decode(), "table_type": TABLE_TYPE_FULLWINDOW_W8,
                   "version": HEADER_VERSION, "header_bytes": 32},
        "layout": "32B typed header, then AoS entry=X[8xu32 LE]||Y[8xu32 LE], index=window*128+(digit-1)",
        "windows": WINDOWS, "digits": DIGITS, "entry_bytes": ENTRY_BYTES,
        "total_bytes": len(blob), "domain": "montgomery (x*R mod p)",
        "sha256": fp, "sha256_covers": "entire file (header+payload)",
        "verified_against": "pyca/cryptography derive_private_key",
        "golden_vectors": goldens,
    }
    with open(bin_path.replace(".bin", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"[SUCCESS] {bin_path}")
    print(f"  size:   {len(blob):,} bytes ({len(blob)/1024:.0f} KB)")
    print(f"  sha256: {fp}")
    print(f"  goldens: {len(goldens)} vectors in meta json")
    print("=" * 72)


if __name__ == "__main__":
    main()
