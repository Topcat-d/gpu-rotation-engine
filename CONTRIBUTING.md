# Developing the standalone engine

Build and run the installed example using the README before changing kernels.
Keep the public C structs and exported function signatures compatible. The native
build uses no Python or PyTorch bindings; Python is only used for asset generation
and independent verification in this SDK.

After a change:

1. Run `python -m pytest tests/test_standalone_tables.py -q`.
2. Rebuild and run the two CTests described in the README.
3. Reinstall and run the full standalone rotation example with
   `--require-fullwindow --report rotation-report.json`.
4. Record the revision, GPU, CUDA version, architecture, and build flags when
   comparing performance. Keep end-to-end correctness checks separate from timed
   measurements, and verify every result against the intended key.

The example measures correctness, not throughput. Preserve nonce uniqueness,
asynchronous fence enforcement, tamper rejection, and previous-key decryption
when optimizing. Coordinate in-flight operations before overwriting key slots.
