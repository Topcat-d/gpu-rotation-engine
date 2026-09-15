#!/usr/bin/env python3
"""
P-256 Fixed-Base Comb Table Generator

Generates Structure-of-Arrays (SoA) comb tables for fixed-base scalar multiplication.
Outputs .npy files ready for cudaMemcpyToSymbol.

Usage:
    python gen_p256_comb_table.py --w 7 --outdir legacy/data/p256_comb
    python gen_p256_comb_table.py --w 8 --outdir legacy/data/p256_comb
"""

import os
import argparse
import numpy as np

# P-256 curve parameters (secp256r1 / NIST P-256)
P = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
A = 0xffffffff00000001000000000000000000000000fffffffffffffffffffffffc  # a = -3
B = 0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
Gx = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296
Gy = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5
N = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551

# Montgomery constant R = 2^256 mod p
R_MOD_P = pow(2, 256, P)
R2_MOD_P = pow(2, 512, P)  # R^2 mod p


def int_to_limbs_le(x: int) -> np.ndarray:
    """Convert integer to 8 little-endian uint32 limbs."""
    return np.array([(x >> (32*i)) & 0xFFFFFFFF for i in range(8)], dtype=np.uint32)


def limbs_le_to_int(le: np.ndarray) -> int:
    """Convert 8 little-endian uint32 limbs to integer."""
    out = 0
    for i, limb in enumerate(le.tolist()):
        out |= int(limb) << (32*i)
    return out


def mod_inverse(a: int, modulus: int) -> int:
    """Modular inverse using Fermat's Little Theorem (for prime modulus)."""
    if a == 0:
        raise ValueError("Cannot compute modular inverse of 0")
    return pow(a, modulus - 2, modulus)


def to_montgomery(x: int) -> int:
    """
    Convert normal field element to Montgomery domain: x -> x*R mod p.

    CARD17 FIX: Use x*R (single Montgomery), NOT x*R^2 (double Montgomery).
    The old buggy formula was: (x * R2_MOD_P) % P = x*R^2
    The correct formula is:    (x * R_MOD_P) % P = x*R
    """
    return (x * R_MOD_P) % P


def from_montgomery(x_mont: int) -> int:
    """Convert Montgomery field element to normal domain."""
    return (x_mont * mod_inverse(R_MOD_P, P)) % P


def point_double_affine(x1: int, y1: int) -> tuple:
    """
    Elliptic curve point doubling in affine coordinates.
    For P-256: y^2 = x^3 + ax + b, where a = -3

    Formula:
    lambda = (3*x1^2 + a) / (2*y1)
    x3 = lambda^2 - 2*x1
    y3 = lambda*(x1 - x3) - y1
    """
    # Handle point at infinity
    if x1 == 0 and y1 == 0:
        return (0, 0)

    # Compute lambda = (3*x1^2 + a) / (2*y1) mod p
    numerator = (3 * x1 * x1 + A) % P
    denominator = (2 * y1) % P
    lambda_val = (numerator * mod_inverse(denominator, P)) % P

    # Compute x3 = lambda^2 - 2*x1
    x3 = (lambda_val * lambda_val - 2 * x1) % P

    # Compute y3 = lambda*(x1 - x3) - y1
    y3 = (lambda_val * (x1 - x3) - y1) % P

    return (x3, y3)


def point_add_affine(x1: int, y1: int, x2: int, y2: int) -> tuple:
    """
    Elliptic curve point addition in affine coordinates.
    P + Q = R

    Formula:
    lambda = (y2 - y1) / (x2 - x1)
    x3 = lambda^2 - x1 - x2
    y3 = lambda*(x1 - x3) - y1
    """
    # Handle special cases
    if x1 == 0 and y1 == 0:  # P is point at infinity
        return (x2, y2)
    if x2 == 0 and y2 == 0:  # Q is point at infinity
        return (x1, y1)
    if x1 == x2:
        if y1 == y2:  # Point doubling
            return point_double_affine(x1, y1)
        else:  # Points are inverses
            return (0, 0)

    # Compute lambda = (y2 - y1) / (x2 - x1) mod p
    numerator = (y2 - y1) % P
    denominator = (x2 - x1) % P
    lambda_val = (numerator * mod_inverse(denominator, P)) % P

    # Compute x3 = lambda^2 - x1 - x2
    x3 = (lambda_val * lambda_val - x1 - x2) % P

    # Compute y3 = lambda*(x1 - x3) - y1
    y3 = (lambda_val * (x1 - x3) - y1) % P

    return (x3, y3)


def gen_signed_window_table(w: int) -> tuple:
    """
    Generate signed fixed-window comb buckets (all multiples of G).

    For window width w:
    - entries = 2^w - 1
    - bucket k corresponds to multiple d = k+1 ∈ {1, 2, 3, ..., 2^w-1}

    Returns:
        X_soa: np.ndarray of shape [8, entries] (x-coordinates in Montgomery domain, SoA layout)
        Y_soa: np.ndarray of shape [8, entries] (y-coordinates in Montgomery domain, SoA layout)
    """
    entries = (1 << w) - 1  # 2^w - 1 (for w=7: 127 entries)

    print(f"Generating comb table for w={w}")
    print(f"  Entries per window: {entries}")
    print(f"  Total points to compute: {entries}")

    # Build all multiples of G: {1G, 2G, 3G, 4G, ...}
    buckets = []

    # 1G = G
    curx, cury = Gx, Gy
    buckets.append((curx, cury))
    print(f"  Computed 1G")

    # We'll add G each time, so no need for 2G
    oneGx, oneGy = Gx, Gy
    print(f"  Using 1G for incrementing")

    # Compute 2G, 3G, 4G, ... by repeatedly adding G
    for i in range(1, entries):
        nx, ny = point_add_affine(curx, cury, oneGx, oneGy)
        buckets.append((nx, ny))
        curx, cury = nx, ny
        if (i + 1) % 8 == 0:
            print(f"  Computed {i+1}G ({i+1}/{entries})")

    print(f"  All {entries} points computed!")

    # Convert to Montgomery domain and Structure-of-Arrays layout
    # SoA shape: [8 limbs, entries]
    X_soa = np.zeros((8, entries), dtype=np.uint32)
    Y_soa = np.zeros((8, entries), dtype=np.uint32)

    print(f"  Converting to Montgomery domain and SoA layout...")
    for idx, (x, y) in enumerate(buckets):
        # Convert to Montgomery domain
        xm = to_montgomery(x)
        ym = to_montgomery(y)

        # Convert to limbs (little-endian)
        X_soa[:, idx] = int_to_limbs_le(xm)
        Y_soa[:, idx] = int_to_limbs_le(ym)

    print(f"  Conversion complete!")

    # Verify first entry (1G) in Montgomery domain
    x1_mont = to_montgomery(Gx)
    y1_mont = to_montgomery(Gy)
    x1_limbs = int_to_limbs_le(x1_mont)
    y1_limbs = int_to_limbs_le(y1_mont)
    assert np.array_equal(X_soa[:, 0], x1_limbs), "X coordinate mismatch for 1G"
    assert np.array_equal(Y_soa[:, 0], y1_limbs), "Y coordinate mismatch for 1G"
    print(f"  Verification: 1G matches!")

    return X_soa, Y_soa


def main():
    parser = argparse.ArgumentParser(
        description="Generate P-256 fixed-base comb tables (SoA layout)"
    )
    parser.add_argument(
        "--w",
        type=int,
        default=7,
        choices=[7, 8],
        help="Window width (7 or 8)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="legacy/data/p256_comb",
        help="Output directory for .npy files"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Generate tables
    print("=" * 80)
    print("P-256 Fixed-Base Comb Table Generator")
    print("=" * 80)
    X_soa, Y_soa = gen_signed_window_table(args.w)

    # Save to .npy files
    x_path = os.path.join(args.outdir, f"p256_G_X_soa_w{args.w}.npy")
    y_path = os.path.join(args.outdir, f"p256_G_Y_soa_w{args.w}.npy")

    np.save(x_path, X_soa)
    np.save(y_path, Y_soa)

    print()
    print("=" * 80)
    print(f"[SUCCESS] Comb tables saved!")
    print(f"  Window width: {args.w}")
    print(f"  X coordinates: {x_path}")
    print(f"  Y coordinates: {y_path}")
    print(f"  Shape: [8 limbs, {X_soa.shape[1]} entries]")
    print(f"  Total size: {(X_soa.nbytes + Y_soa.nbytes) / 1024:.2f} KB")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Use cudaMemcpyToSymbol to upload these tables to GPU")
    print("  2. Update your kernel to use the correct window width")
    print("  3. Test with correctness harness")
    print()


if __name__ == "__main__":
    main()
