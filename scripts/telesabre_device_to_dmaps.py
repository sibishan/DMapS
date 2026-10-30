#!/usr/bin/env python3
"""Convert a telesabre device JSON into a DMapS chip-network JSON.

Telesabre assigns qubit q to core q // (num_qubits / num_cores). Each core
becomes a DMapS chip, each intra-core edge a coupling and each inter-core edge
a remote connection. Qubits are named "Q<telesabre index>", and DMapS numbers
qubits in the order they are listed, so DMapS indices match telesabre's.

Usage:
    python scripts/telesabre_device_to_dmaps.py <device.json> [-o out.json]
        [--intra-fid 0.99] [--remote-fid 0.9] [--single-fid 0.999]
"""
import argparse
import json
from pathlib import Path


def convert(device, intra_fid, remote_fid, single_fid):
    num_qubits = device["num_qubits"]
    num_cores = device["num_cores"]
    if num_qubits % num_cores:
        raise ValueError("num_qubits must be divisible by num_cores")
    capacity = num_qubits // num_cores

    def core(q):
        return q // capacity

    chips = {}
    for c in range(num_cores):
        qubits = list(range(c * capacity, (c + 1) * capacity))
        chips[f"chip {c}"] = {
            "use calibration file": False,
            "qubits": [f"Q{q}" for q in qubits],
            "fidelity": {f"Q{q}": {"x/2": single_fid} for q in qubits},
            "couplings": [],
        }

    for u, v in device["intra_core_edges"]:
        if core(u) != core(v):
            raise ValueError(f"intra-core edge ({u}, {v}) spans cores")
        chips[f"chip {core(u)}"]["couplings"].append(
            {"qubit pair": [f"Q{u}", f"Q{v}"], "fidelity": intra_fid}
        )

    remote = []
    for u, v in device["inter_core_edges"]:
        if core(u) == core(v):
            raise ValueError(f"inter-core edge ({u}, {v}) is inside one core")
        remote.append(
            {f"chip {core(u)}": f"Q{u}", f"chip {core(v)}": f"Q{v}", "fidelity": remote_fid}
        )

    return {"has multiple chips": True, "chips": chips, "remote connection": remote}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("device", type=Path, help="telesabre device JSON")
    parser.add_argument("-o", "--output", type=Path, help="output path (default: tests/hardware_info/telesabre/<name>.json)")
    parser.add_argument("--intra-fid", type=float, default=0.99, help="fidelity of intra-core couplings")
    parser.add_argument("--remote-fid", type=float, default=0.9, help="fidelity of inter-core links")
    parser.add_argument("--single-fid", type=float, default=0.999, help="single-qubit gate fidelity")
    args = parser.parse_args()

    device = json.loads(args.device.read_text())["device"]
    out = args.output or Path(__file__).resolve().parent.parent / "tests" / "hardware_info" / "telesabre" / f"{args.device.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert(device, args.intra_fid, args.remote_fid, args.single_fid), indent=2))
    print(f"Wrote {out}")
