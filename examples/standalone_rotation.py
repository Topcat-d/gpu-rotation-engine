#!/usr/bin/env python3
"""Verify an installed Smoke C ABI: tables, P-256, and AEAD key rotation.

Requires numpy and cryptography as independent test oracles, not PyTorch.
This is a sequential integration test, not a production key-management service.
Keys are freshly generated in memory and never printed or written to the report.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import faulthandler
import json
import os
from pathlib import Path
import secrets
import sys
import time

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305


U8, U32, U64, PTR = C.c_uint8, C.c_uint32, C.c_uint64, C.c_void_p


class Config(C.Structure):
    _fields_ = [("version", U32), ("device", C.c_int32), ("queue", U32),
                ("reserved32", U32 * 4), ("reserved64", U64 * 4)]


class Error(C.Structure):
    _fields_ = [("code", C.c_int), ("internal", C.c_int32), ("message", C.c_char * 256)]


class Request(C.Structure):
    _fields_ = [("version", U32), ("opcode", C.c_uint16), ("flags", U8),
                ("slot", U8), ("client", U64), ("length", C.c_uint16),
                ("reserved16", C.c_uint16), ("offset", U32), ("payload_len", U32),
                ("reserved32", U32), ("input", U8 * 32)]


class Response(C.Structure):
    _fields_ = [("client", U64), ("seq", U64), ("status", U8), ("backend", U8),
                ("opcode", C.c_uint16), ("length", C.c_uint16), ("reserved16", C.c_uint16),
                ("offset", U32), ("output_bytes", U32), ("submit", U64),
                ("dequeue", U64), ("complete", U64), ("request_id", U64),
                ("output", U8 * 64)]


class SlotProbe(C.Structure):
    _fields_ = [("slot", U32), ("epoch", U32), ("key_type", U32),
                ("key_len", C.c_uint16), ("pad", C.c_uint16),
                ("checks", U32 * 9), ("top_bytes", U8 * 4),
                ("faults", U32), ("version", U32), ("healthy", U32)]


class Probe(C.Structure):
    _fields_ = [("active", U32), ("num_slots", U32), ("healthy", U32),
                ("error", U32), ("slots", SlotProbe * 2), ("checks", U32 * 3),
                ("message", C.c_char * 128)]


class KeyId(C.Structure):
    _fields_ = [("id", U32), ("reserved", U32)]


class P256Key(C.Structure):
    _fields_ = [("d", U8 * 32), ("reserved", U8 * 32)]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def buffer(data):
    return (U8 * len(data)).from_buffer_copy(data)


class Engine:
    def __init__(self, library: Path, device: int):
        require(C.sizeof(Request) == 64 and C.sizeof(Response) == 128, "ABI layout mismatch")
        self.dll_dirs = []
        if os.name == "nt":
            for directory in (library.parent, Path(os.environ.get("CUDA_PATH", "")) / "bin"):
                if directory.is_dir():
                    self.dll_dirs.append(os.add_dll_directory(str(directory.resolve())))
        self.lib = C.CDLL(str(library.resolve()))
        self.handle = PTR()
        self.started = False
        self.operations = 0
        self.nonce_counter = 0
        self.nonce_prefix = secrets.token_bytes(4)
        self._bind()
        cfg, err = Config(version=1, device=device, queue=1024), Error()
        print("Initializing native engine", file=sys.stderr, flush=True)
        self.ok(self.lib.smoke_engine_init(C.byref(cfg), C.byref(self.handle), C.byref(err)), "init", err)

    def _bind(self):
        def bind(name, args, result=C.c_int):
            fn = getattr(self.lib, "smoke_" + name)
            fn.argtypes, fn.restype = args, result
        pe = C.POINTER(Error)
        bind("engine_init", [C.POINTER(Config), C.POINTER(PTR), pe])
        for name in ("engine_shutdown", "fast_path_stop", "engine_commit_rotation",
                     "engine_commit_rotation_async", "engine_rollback_rotation", "engine_rotation_fence"):
            bind(name, [PTR])
        for name in ("fast_path_create", "fast_path_start"):
            bind(name, [PTR, pe])
        for name in ("fast_path_set_num_ctas", "fast_path_set_num_shards"):
            bind(name, [PTR, U32])
        bind("engine_load_key_aes256", [PTR, U32, PTR, pe])
        bind("engine_load_key_p256", [PTR, C.POINTER(KeyId), C.POINTER(P256Key)])
        for name in ("engine_prepare_next_key_typed", "engine_prepare_next_key_typed_async"):
            bind(name, [PTR, U32, PTR, U32])
        bind("engine_rotation_probe", [PTR, C.POINTER(Probe)])
        bind("engine_load_p256_comb_table", [PTR, PTR, PTR, C.c_size_t])
        bind("engine_load_p256_fullwindow_table", [PTR, PTR, C.c_size_t, C.c_char_p])
        bind("engine_get_payload_slab", [PTR, C.POINTER(PTR), C.POINTER(U32)])
        bind("generic_submit", [PTR, C.POINTER(Request), U32, pe])
        bind("generic_poll", [PTR, C.POINTER(Response), U32], U32)
        bind("response_copy_output", [PTR, C.POINTER(Response), PTR, C.POINTER(U32)])

    @staticmethod
    def ok(status, label, error=None):
        detail = bytes(error.message).decode(errors="replace") if error else ""
        require(status == 0, f"{label} failed: status={status} {detail}")

    def load_tables(self, data_dir, require_fullwindow):
        print("Loading and checking tables", file=sys.stderr, flush=True)
        tables = [np.ascontiguousarray(np.load(data_dir / "p256_comb" / f"p256_G_{axis}_soa_w8.npy",
                                             allow_pickle=False)) for axis in ("X", "Y")]
        for table in tables:
            require(table.shape == (8, 255) and table.dtype == np.dtype("uint32"), "Invalid comb table")
        # init starts the legacy kernel. Table upload must fail promptly until
        # FAST PATH creation stops it; default-stream copies would otherwise hang.
        status = self.lib.smoke_engine_load_p256_comb_table(self.handle, tables[0].ctypes.data,
                                                           tables[1].ctypes.data, 2040)
        require(status != 0, "Table loading while legacy kernel runs was accepted")
        self.ok(self.lib.smoke_fast_path_create(self.handle, None), "create fast path")
        self.ok(self.lib.smoke_engine_load_p256_comb_table(self.handle, tables[0].ctypes.data,
                                                          tables[1].ctypes.data, 2040), "load comb tables")
        blob = (data_dir / "p256_fullwindow" / "p256_G_fullwindow_w8.bin").read_bytes()
        bad = bytearray(blob)
        bad[-1] ^= 1
        status = self.lib.smoke_engine_load_p256_fullwindow_table(self.handle, buffer(bad), len(bad), None)
        require(status != 0, "Corrupted full-window table was accepted")
        status = self.lib.smoke_engine_load_p256_fullwindow_table(self.handle, buffer(blob), len(blob), None)
        if require_fullwindow:
            self.ok(status, "load full-window table")
        else:
            require(status in (0, 7), f"Unexpected table loader status {status}")
        return {"comb": "w8", "fullwindow_enabled": status == 0,
                "fullwindow_sha256": hashlib.sha256(blob).hexdigest()}

    def start(self):
        print("Starting persistent GPU kernel", file=sys.stderr, flush=True)
        err = Error()
        self.ok(self.lib.smoke_fast_path_set_num_ctas(self.handle, 2), "set CTAs")
        self.ok(self.lib.smoke_fast_path_set_num_shards(self.handle, 2), "set shards")
        self.ok(self.lib.smoke_fast_path_start(self.handle, C.byref(err)), "start", err)
        self.started = True
        self.payload, self.payload_size = PTR(), U32()
        self.ok(self.lib.smoke_engine_get_payload_slab(self.handle, C.byref(self.payload),
                                                       C.byref(self.payload_size)), "payload slab")

    def probe(self):
        result = Probe()
        self.ok(self.lib.smoke_engine_rotation_probe(self.handle, C.byref(result)), "rotation probe")
        require(result.error == 0 and result.active in (0, 1), "Invalid epoch probe")
        return result

    def load_symmetric(self, slot, key):
        self.ok(self.lib.smoke_engine_load_key_aes256(self.handle, slot, buffer(key), None), "load symmetric key")

    def prepare(self, key, key_type=2, asynchronous=False):
        name = "smoke_engine_prepare_next_key_typed" + ("_async" if asynchronous else "")
        self.ok(getattr(self.lib, name)(self.handle, key_type, buffer(key), len(key)), "prepare key")

    def nonce(self):
        self.nonce_counter += 1
        return self.nonce_prefix + self.nonce_counter.to_bytes(8, "big")

    def request(self, opcode, slot, inline, payload=None, expected_submit=0):
        req = Request(version=1, opcode=opcode, slot=slot, length=len(inline))
        require(len(inline) <= 32, "Inline request too large")
        req.input[:len(inline)] = inline
        if payload is not None:
            require(0 < len(payload) <= self.payload_size.value - 64, "Invalid payload length")
            req.offset, req.payload_len = 64, len(payload)
            # Sequential harness: only one request owns this payload region at a time.
            C.memmove(self.payload.value + 64, payload, len(payload))
        err = Error()
        status = self.lib.smoke_generic_submit(self.handle, C.byref(req), 1, C.byref(err))
        require(status == expected_submit, f"submit opcode {opcode}: {status}, expected {expected_submit}: {bytes(err.message)!r}")
        if expected_submit:
            return None, b""
        result, deadline = Response(), time.monotonic() + 10
        while self.lib.smoke_generic_poll(self.handle, C.byref(result), 1) == 0:
            require(time.monotonic() < deadline, f"Timeout waiting for opcode {opcode}")
            time.sleep(0.0001)
        require(result.backend == 1, "Operation did not execute through GPU fast path")
        self.operations += 1
        if result.status:
            require(result.length == 0 and result.output_bytes == 0, "Failed operation exposed output")
            return result.status, b""
        output, size = (U8 * 65552)(), U32(65552)
        self.ok(self.lib.smoke_response_copy_output(self.handle, C.byref(result), output, C.byref(size)), "copy output")
        return 0, bytes(output[:size.value])

    def aead(self, opcode, slot, nonce, body):
        # Seal inline is limited to 16 plaintext bytes; open inline to 4 ciphertext bytes.
        inline_limit = 16 if opcode in (20, 21) else 20
        return self.request(opcode, slot, nonce + body) if len(body) <= inline_limit else self.request(opcode, slot, nonce, body)

    def close(self):
        if self.handle:
            if self.started:
                self.ok(self.lib.smoke_fast_path_stop(self.handle), "stop")
            self.ok(self.lib.smoke_engine_shutdown(self.handle), "shutdown")
            self.handle = PTR()
        for directory in self.dll_dirs:
            directory.close()


def verify_p256(engine):
    key = ec.generate_private_key(ec.SECP256R1())
    native = P256Key()
    native.d[:] = key.private_numbers().private_value.to_bytes(32, "big")
    engine.ok(engine.lib.smoke_engine_load_key_p256(engine.handle, C.byref(KeyId()), C.byref(native)), "load P-256")
    engine.start()
    engine.ok(engine.lib.smoke_engine_load_key_p256(engine.handle, C.byref(KeyId()), C.byref(native)), "reload P-256 after start")
    for generation in range(3):
        print(f"Verifying P-256 generation {generation}", file=sys.stderr, flush=True)
        if generation:
            key = ec.generate_private_key(ec.SECP256R1())
            engine.prepare(key.private_numbers().private_value.to_bytes(32, "big"), key_type=1)
            engine.ok(engine.lib.smoke_engine_commit_rotation(engine.handle), "commit P-256")
        slot = engine.probe().active
        for sample in range(4):
            digest = hashlib.sha256(f"standalone:{generation}:{sample}".encode()).digest()
            status, signature = engine.request(0, slot, digest)
            require(status == 0 and len(signature) == 64, "P-256 signing failed")
            der = utils.encode_dss_signature(int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big"))
            key.public_key().verify(der, digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
    return {"signatures_verified": 12, "rotations": 2}


def verify_aead(engine, rounds):
    # AES and ChaCha share the symmetric slot representation. Each generation
    # uses a new random key, and every encryption gets a unique 96-bit nonce.
    active = engine.probe().active
    key = secrets.token_bytes(32)
    engine.load_symmetric(active, key)
    previous = None
    archived = None
    lengths = (0, 1, 4, 16, 17, 20, 32, 64, 255, 1024, 4096, 65535)
    checks = 0
    for generation in range(rounds + 1):
        print(f"Verifying AEAD generation {generation}", file=sys.stderr, flush=True)
        if generation:
            previous_slot, previous_key, previous_records = active, key, records
            next_key = secrets.token_bytes(32)
            # Prepare/rollback must preserve the active key and slot.
            engine.prepare(secrets.token_bytes(32))
            engine.ok(engine.lib.smoke_engine_rollback_rotation(engine.handle), "rollback")
            require(engine.probe().active == active, "Rollback changed active slot")
            rollback_nonce = engine.nonce()
            status, rollback_ct = engine.aead(20, active, rollback_nonce, b"rollback")
            require(status == 0 and rollback_ct == AESGCM(key).encrypt(rollback_nonce, b"rollback", None),
                    "Rollback changed the active encryption key")
            if generation % 2:
                engine.prepare(next_key, asynchronous=True)
                engine.request(20, active, engine.nonce(), expected_submit=10)
                engine.ok(engine.lib.smoke_engine_commit_rotation_async(engine.handle), "async commit")
                engine.request(20, active ^ 1, engine.nonce(), expected_submit=10)
                engine.ok(engine.lib.smoke_engine_rotation_fence(engine.handle), "rotation fence")
            else:
                engine.prepare(next_key)
                engine.ok(engine.lib.smoke_engine_commit_rotation(engine.handle), "sync commit")
            active = engine.probe().active
            require(active == (previous_slot ^ 1), "Commit did not select the next slot")
            key = next_key
            previous = (previous_slot, previous_key, previous_records)
        records = []
        for seal, open_op, oracle_cls in ((20, 24, AESGCM), (21, 25, ChaCha20Poly1305)):
            oracle = oracle_cls(key)
            for size in lengths:
                plaintext = bytes((i * 17 + generation) % 256 for i in range(size))
                nonce = engine.nonce()
                status, ciphertext = engine.aead(seal, active, nonce, plaintext)
                require(status == 0, f"seal opcode={seal} size={size} status={status}")
                require(ciphertext == oracle.encrypt(nonce, plaintext, None), f"Ciphertext mismatch: opcode={seal} size={size}")
                status, recovered = engine.aead(open_op, active, nonce, ciphertext)
                require(status == 0 and recovered == plaintext, f"GPU decrypt mismatch: opcode={open_op} size={size}")
                require(oracle.decrypt(nonce, ciphertext, None) == plaintext, "CPU decrypt mismatch")
                tampered = ciphertext[:-1] + bytes([ciphertext[-1] ^ 1])
                status, rejected = engine.aead(open_op, active, nonce, tampered)
                require(status != 0 and rejected == b"", "Tampered ciphertext accepted")
                records.append((open_op, nonce, ciphertext, plaintext))
                checks += 1
        if generation == 0:
            # A real application keeps historical keys in its own secure key
            # manager, indexed by immutable key ID, not by a reused GPU slot.
            archived = (key, list(records))
        if previous:
            old_slot, old_key, old_records = previous
            for open_op, nonce, ciphertext, plaintext in old_records:
                status, recovered = engine.aead(open_op, old_slot, nonce, ciphertext)
                require(status == 0 and recovered == plaintext, "Previous epoch ciphertext no longer decrypts")
                status, rejected = engine.aead(open_op, active, nonce, ciphertext)
                require(status != 0 and rejected == b"", "New key accepted old ciphertext")
    archived_key, archived_records = archived
    restore_slot = active ^ 1
    engine.load_symmetric(restore_slot, archived_key)
    for open_op, nonce, ciphertext, plaintext in archived_records:
        status, recovered = engine.aead(open_op, restore_slot, nonce, ciphertext)
        require(status == 0 and recovered == plaintext, "Archived-key restore failed")
    # Restoring an old key in the inactive slot must not disturb the current one.
    for open_op, nonce, ciphertext, plaintext in records:
        status, recovered = engine.aead(open_op, active, nonce, ciphertext)
        require(status == 0 and recovered == plaintext, "Archived-key restore damaged active key")
    return {"algorithms": ["AES-256-GCM", "ChaCha20-Poly1305"], "rotations": rounds,
            "sync_rotations": rounds // 2, "async_rotations": (rounds + 1) // 2,
            "roundtrips_verified": checks, "plaintext_sizes": list(lengths),
            "tamper_rejection": True, "wrong_key_rejection": True,
            "previous_epoch_decryption": True, "async_fence_enforced": True,
            "archived_records_restored": len(archived_records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--require-fullwindow", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(120, exit=True)
    require(args.rotations >= 2, "Use at least two rotations to exercise both slots")
    engine = Engine(args.library, args.device)
    try:
        tables = engine.load_tables(args.data_dir, args.require_fullwindow)
        signing = verify_p256(engine)
        aead = verify_aead(engine, args.rotations)
        report = {"status": "passed", "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  "device_index": args.device, "tables": tables, "p256": signing,
                  "aead": aead, "gpu_operations": engine.operations,
                  "scope": "sequential installed C ABI validation; no AAD; not a throughput or concurrency claim"}
    finally:
        engine.close()
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
