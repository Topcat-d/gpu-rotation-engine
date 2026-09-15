"""Shipped precomputation assets must match their reproducible generators."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("width", [7, 8])
def test_shipped_comb_tables_match_generator(tmp_path, width):
    subprocess.run([sys.executable, str(ROOT / "scripts/tools/gen_p256_comb_table.py"),
                    "--w", str(width), "--outdir", str(tmp_path)], check=True, capture_output=True)
    for axis in ("X", "Y"):
        name = f"p256_G_{axis}_soa_w{width}.npy"
        shipped = np.load(ROOT / "data/p256_comb" / name, allow_pickle=False)
        expected = np.load(tmp_path / name, allow_pickle=False)
        assert shipped.dtype == np.dtype("uint32")
        assert shipped.shape == (8, (1 << width) - 1)
        np.testing.assert_array_equal(shipped, expected)


def test_fullwindow_matches_independently_verified_generator(tmp_path):
    # The generator cross-checks EVERY point against pyca/cryptography. Do not
    # pass --skip-pyca: comparing two copies of our own curve math is insufficient.
    subprocess.run([sys.executable, str(ROOT / "scripts/tools/gen_p256_fullwindow_table.py"),
                    "--outdir", str(tmp_path)], check=True, capture_output=True)
    name = "p256_G_fullwindow_w8.bin"
    assert (ROOT / "data/p256_fullwindow" / name).read_bytes() == (tmp_path / name).read_bytes()
