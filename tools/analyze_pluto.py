#!/usr/bin/env python3
"""Read-only analysis of the pinned Pluto CoG 2026 release; never executes PE files.

Requires pefile==2024.8.26. Output contains metadata, not model weights.
"""

import argparse
import collections
import hashlib
import json
import math
import re
import struct
import zipfile
from pathlib import Path

ARCHIVE_SHA = "d4e2225446f5048e131065357173952f0286586da77ebb8bddf7fc766c3844a5"
EXPECTED = {
    "pluto.dll": "7e360b643c8c0156c03fe0cad9972a3058138ccfe22f921c5b4e0cd0aaf0abef",
    "pluto/pluto_infer.exe": "ad880d8be52a6ef03893fa20644b6627e04fcd55e030178c6d52486b82340f2b",
    "pluto/pluto_weights.bin": "00b400eace6e4782202ebdcb3c30db76054aaa6a08c6a7dcb59575abc2d0a26e",
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def weights(path):
    with path.open("rb") as f:
        magic, version, length = struct.unpack("<4sIQ", f.read(16))
        if magic != b"PLWB" or version != 1 or not 0 < length <= 1024 * 1024:
            raise ValueError("Unsupported weight header")
        header = json.loads(f.read(length))
        payload = (16 + length + 63) // 64 * 64
        if any(f.read(payload - 16 - length)):
            raise ValueError("Nonzero header alignment padding")
    if path.stat().st_size != payload + header["total_bytes"]:
        raise ValueError("Weight payload length mismatch")
    types = {"float32": 4, "bfloat16": 2, "int8": 1}
    groups, counts, parameters = collections.Counter(), collections.Counter(), collections.Counter()
    spans, scales = [], 0
    for tensor in header["tensors"]:
        name, shape = tensor["name"], tensor["shape"]
        if not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("Invalid tensor shape")
        dtype = tensor.get("dtype", header["dtype"])
        n = math.prod(shape)
        if n * types[dtype] != tensor["nbytes"]:
            raise ValueError("Incorrect storage size: " + name)
        groups[name.split(".")[0]] += n
        counts[dtype] += 1
        parameters[dtype] += n
        spans.append((tensor["offset"], tensor["offset"] + tensor["nbytes"], name))
        if dtype == "int8":
            start, count = tensor["scale_offset"], tensor["scale_len"]
            scales += count
            spans.append((start, start + count * 4, name + ".scales"))
    spans.sort()
    previous, gaps = 0, 0
    for start, end, name in spans:
        if start < previous or end > header["total_bytes"] or end < start:
            raise ValueError("Overlapping or invalid tensor extent: " + name)
        gaps += start - previous
        previous = end
    if previous != header["total_bytes"]:
        raise ValueError("Unaccounted payload tail")
    scale_values = []
    with path.open("rb") as f:
        for t in header["tensors"]:
            if t.get("dtype") == "int8":
                f.seek(payload + t["scale_offset"])
                scale_values.extend(
                    struct.unpack("<" + "f" * t["scale_len"], f.read(4 * t["scale_len"]))
                )
    if not all(math.isfinite(v) and v > 0 for v in scale_values):
        raise ValueError("Non-finite or non-positive int8 scales")
    return header, {
        "magic": magic.decode(),
        "version": version,
        "jsonBytes": length,
        "payloadOffset": payload,
        "payloadBytes": header["total_bytes"],
        "tensorCount": len(header["tensors"]),
        "parameters": sum(parameters.values()),
        "parametersByDtype": dict(parameters),
        "tensorCountByDtype": dict(counts),
        "parametersByGroup": dict(groups.most_common()),
        "scaleFloatCount": scales,
        "scaleMin": min(scale_values),
        "scaleMax": max(scale_values),
        "paddingBytes": gaps,
        "tensorRangesValidated": True,
        "config": header["model_config"],
        "strippedPrefixes": header["stripped_prefixes"],
    }


def binary(path):
    import pefile

    pe = pefile.PE(str(path))
    raw = path.read_bytes()
    imports = {
        d.dll.decode(): [s.name.decode() if s.name else f"ordinal:{s.ordinal}" for s in d.imports]
        for d in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
    }
    exports = [
        (s.name or b"").decode()
        for s in getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", [])
    ]
    needles = (
        "BWRL_",
        "PLUTO_SHM_",
        "latency",
        "retained observation",
        "frames_per_step",
        "handshake",
        "bandit",
        "win_final",
        "int8 blob",
        "AVX2",
        "arch_v2",
        "pluto_config.json",
        "no fp32 materialization",
    )
    evidence = []
    for match in re.finditer(rb"[\x20-\x7e]{6,}", raw):
        value = match.group().decode()
        if any(n in value for n in needles):
            evidence.append({"offset": match.start(), "text": value})
    return {
        "machine": hex(pe.FILE_HEADER.Machine),
        "optionalHeaderMagic": hex(pe.OPTIONAL_HEADER.Magic),
        "imageBase": hex(pe.OPTIONAL_HEADER.ImageBase),
        "entryRva": hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint),
        "sections": [
            {
                "name": s.Name.rstrip(b"\0").decode(),
                "rva": s.VirtualAddress,
                "rawSize": s.SizeOfRawData,
            }
            for s in pe.sections
        ],
        "coffSymbols": pe.FILE_HEADER.NumberOfSymbols,
        "imports": imports,
        "exports": exports,
        "selectedStrings": evidence,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--extract", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if digest(args.archive) != ARCHIVE_SHA:
        raise ValueError("Archive does not match pinned official SHA256")
    args.extract.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        if z.testzip() is not None:
            raise ValueError("Bad ZIP CRC")
        members = [
            {"name": i.filename, "bytes": i.file_size, "compressedBytes": i.compress_size}
            for i in z.infolist()
        ]
        # Extract only exact known release member names; no archive-defined paths.
        for name in (*EXPECTED, "README.txt"):
            if z.getinfo(name).file_size > 512 * 1024 * 1024:
                raise ValueError("Unexpected member size")
            dest = args.extract / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
    files = {}
    for name, expected in EXPECTED.items():
        p = args.extract / name
        actual = digest(p)
        if actual != expected:
            raise ValueError("Member hash mismatch: " + name)
        files[name] = {"bytes": p.stat().st_size, "sha256": actual}
    header, summary = weights(args.extract / "pluto/pluto_weights.bin")
    write(
        args.out / "manifest.json",
        {
            "release": "cog2026-2578600",
            "archiveSha256": ARCHIVE_SHA,
            "members": members,
            "verifiedFiles": files,
        },
    )
    write(args.out / "weights-header.json", header)
    write(args.out / "model-summary.json", summary)
    write(
        args.out / "binaries.json",
        {name: binary(args.extract / name) for name in EXPECTED if name.endswith((".exe", ".dll"))},
    )
    print(
        json.dumps(
            {
                "hashesVerified": 4,
                "parameters": summary["parameters"],
                "tensors": summary["tensorCount"],
                "output": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
