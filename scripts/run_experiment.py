#!/usr/bin/env python3
"""Run DMapS (CPA mapping + CAR routing) over benchmark circuits and seeds.

For every circuit in each batch directory, DMapS is run once per seed and the
run with the fewest EPR pairs is kept as the best. All runs plus the best are
written to a single JSON file, in the same layout as reference/boltsabre.json
(without the operation logs).

Each batch is paired with a device, matching the telesabre/boltsabre
experiments: qasm_25 -> G, qasm_64 -> H (DMapS versions of the telesabre
devices, made by scripts/telesabre_device_to_dmaps.py).

EPR accounting: DMapS routes over the chip network as a plain coupling graph,
so every inter-chip operation in the routed circuit sits on a remote link.
  - a CX on a remote link is a telegate: 1 EPR pair
  - a SWAP on a remote link exchanges two data qubits between chips: counted
    as two teleportations, i.e. --remote-swap-epr (default 2) EPR pairs

Seeding: DMapS is not reproducible out of the box. This script seeds the
two-phase mapper (Python `random`), the KaHyPar partitioner (its config uses
seed=-1) and the local SABRE mapper, runs the mappers' Pool jobs in-process,
and fixes two orderings that varied between processes, so that a seed always
gives the same result.

Usage:
    python scripts/run_experiment.py [--num-seeds 10] [--start-seed 0]
        [--batch qasm_25:G --batch qasm_64:H] [--circuit qft] [-o out.json]
"""
import argparse
import contextlib
import json
import os
import random
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import kahypar  # noqa: E402
from qiskit import QuantumCircuit, QuantumRegister  # noqa: E402

import mapper.intra_processor_mapper  # noqa: E402
import mapper.local_sabre_mapper  # noqa: E402
import mapper.two_phase_mapper  # noqa: E402
import router.car_layout_routing  # noqa: E402
from frontend.chips_info_reader import QuHardwareInfoReader, ScChip  # noqa: E402
from router.multi_mode_routing import MultiModeRouting  # noqa: E402

# Chips are kept in a set and hashed by object id, so their iteration order
# (and hence the mapping) changed from process to process. Hash by chip index.
ScChip.__hash__ = lambda self: hash(self._chip_idx)


class _SerialPool:
    """Stand-in for multiprocessing.Pool that runs jobs in this process.

    The mappers farm their search out to a Pool whose (spawned) workers draw
    from unseeded `random`, so runs could not be reproduced. Running the jobs
    here makes them use the run's seed; seeds run in parallel instead (--jobs).
    """

    def __init__(self, *args, **kwargs):
        pass

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def close(self):
        pass

    def join(self):
        pass


mapper.two_phase_mapper.Pool = _SerialPool
mapper.intra_processor_mapper.Pool = _SerialPool


def _bit_key(bit):
    return (bit.register.name, bit.index)


def _sorted_swaps(obtain_swaps):
    """Return candidate swaps in a fixed order instead of set order.

    Both routers break score ties by candidate order, and set order varied
    between processes, so the same seed could pick different swaps.
    """
    def wrapper(*args, **kwargs):
        return sorted(obtain_swaps(*args, **kwargs), key=lambda s: (_bit_key(s[0]), _bit_key(s[1])))
    return wrapper


_multi_mode_swap_init = mapper.local_sabre_mapper.MultiModeSwap.__init__


def _seeded_multi_mode_swap_init(self, *args, **kwargs):
    """MultiModeSwap breaks ties with default_rng(None), i.e. OS entropy.
    Draw its seed from numpy's global generator, which run_once seeds."""
    _multi_mode_swap_init(self, *args, **kwargs)
    if self.seed is None:
        self.seed = np.random.randint(0, np.iinfo(np.int32).max)


mapper.local_sabre_mapper.MultiModeSwap.__init__ = _seeded_multi_mode_swap_init
mapper.local_sabre_mapper.MultiModeSwap._obtain_swaps = _sorted_swaps(
    mapper.local_sabre_mapper.MultiModeSwap._obtain_swaps)
router.car_layout_routing.CARSwap._obtain_swaps = _sorted_swaps(
    router.car_layout_routing.CARSwap._obtain_swaps)

CIRCUITS_DIR = REPO / "experiments" / "circuits"
DEVICES_DIR = REPO / "experiments" / "devices"
RESULTS_DIR = REPO / "experiments" / "results"
CHIP_TYPE = "ibm"  # only used when a device reads calibration files

_kahypar_partition = kahypar.partition
_kahypar_seed = None


def _seeded_partition(hypergraph, context):
    context.setSeed(_kahypar_seed)
    return _kahypar_partition(hypergraph, context)


kahypar.partition = _seeded_partition


@contextlib.contextmanager
def silenced():
    """Silence stdout/stderr at the fd level (DMapS and KaHyPar print a lot)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def count_ops(circuit: QuantumCircuit, qubit_chip: dict, remote_swap_epr: int) -> dict:
    """Tally local/remote two-qubit operations of a routed circuit."""
    counts = {"remote_cx": 0, "remote_swaps": 0, "swaps": 0}
    for instr in circuit.data:
        name = instr.operation.name
        if name not in ("cx", "swap"):
            continue
        q0, q1 = (circuit.find_bit(q).index for q in instr.qubits)
        remote = qubit_chip[q0] != qubit_chip[q1]
        if name == "cx" and remote:
            counts["remote_cx"] += 1
        elif name == "swap":
            counts["remote_swaps" if remote else "swaps"] += 1
    counts["epr"] = counts["remote_cx"] + remote_swap_epr * counts["remote_swaps"]
    return counts


def single_register_qasm(qasm_path: Path, tmp_dir: str) -> Path:
    """Return a QASM file whose qubits are all in one register named "q".

    DMapS re-indexes qubits by their index within their own register, so a
    circuit with several quantum registers (e.g. ae: eval[24] + q[1]) maps
    different qubits onto the same index. Merge the registers in order.
    """
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    if len(circuit.qregs) == 1:
        return qasm_path
    flat = QuantumCircuit(QuantumRegister(circuit.num_qubits, "q"), *circuit.cregs,
                          name=circuit.name, global_phase=circuit.global_phase)
    flat.compose(circuit, qubits=flat.qubits, clbits=flat.clbits, inplace=True)
    out = Path(tmp_dir) / qasm_path.name
    flat.qasm(filename=str(out))
    return out


def run_once(qasm_path: Path, device_path: Path, qubit_chip: dict, seed: int, remote_swap_epr: int) -> dict:
    global _kahypar_seed
    random.seed(seed)
    np.random.seed(seed)
    _kahypar_seed = seed

    start = time.time()
    try:
        compiler = MultiModeRouting()
        with tempfile.TemporaryDirectory() as tmp_dir, silenced():
            _, _, routed = compiler.cpa_map_car_rout(
                origin_qasm_fn=single_register_qasm(qasm_path, tmp_dir),
                chip_info_fn=device_path,
                chip_type=CHIP_TYPE,
                intra_chip_all2all=False,
            )
    except Exception as exc:  # keep going; record the failure
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}
    return {
        "seed": seed,
        **count_ops(routed, qubit_chip, remote_swap_epr),
        "circuit_depth": routed.depth(),
        "mapping_time": round(compiler.mapping_time, 3),
        "routing_time": round(compiler.routing_time, 3),
        "total_time": round(time.time() - start, 3),
    }


def run_file(qasm_path: Path, device_path: Path, qubit_chip: dict, seeds: list,
             remote_swap_epr: int, executor) -> dict:
    futures = [executor.submit(run_once, qasm_path, device_path, qubit_chip, seed, remote_swap_epr)
               for seed in seeds]
    runs = []
    for future in futures:
        run = future.result()
        if "error" in run:
            print(f"    seed {run['seed']}: FAILED {run['error']}", flush=True)
        else:
            print(f"    seed {run['seed']}: epr={run['epr']} (remote cx={run['remote_cx']}, "
                  f"remote swaps={run['remote_swaps']}) swaps={run['swaps']} "
                  f"depth={run['circuit_depth']}  [{run['total_time']:.1f}s]", flush=True)
        runs.append(run)

    valid = [r for r in runs if "error" not in r]
    best = min(valid, key=lambda r: (r["epr"], r["swaps"], r["seed"])) if valid else None
    return {
        "file": qasm_path.name,
        "path": str(qasm_path.relative_to(REPO)),
        "num_seeds": len(seeds),
        "best_seed": best["seed"] if best else None,
        "best": {k: best[k] for k in ("seed", "epr", "remote_cx", "remote_swaps", "swaps", "circuit_depth")}
        if best else None,
        "num_valid_runs": len(valid),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--batch", action="append", metavar="DIR:DEVICE",
                        help="circuit dir under experiments/circuits and device name (default: qasm_25:G and qasm_64:H)")
    parser.add_argument("--circuit", action="append",
                        help="only run circuits whose file name contains this (repeatable)")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                        help="seeds run in parallel (default: number of CPUs)")
    parser.add_argument("--remote-swap-epr", type=int, default=2, help="EPR pairs charged per remote SWAP")
    parser.add_argument("-o", "--output", type=Path,
                        help="output JSON (default: experiments/results/dmaps_<timestamp>.json)")
    args = parser.parse_args()

    batches = [b.split(":") for b in (args.batch or ["qasm_25:G", "qasm_64:H"])]
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    out = args.output or RESULTS_DIR / f"dmaps_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "method": "DMapS cpa_map_car_rout",
        "num_seeds": args.num_seeds,
        "seeds": seeds,
        "remote_swap_epr": args.remote_swap_epr,
        "batches": [],
    }

    # One fresh process per run, so no state carries over between seeds.
    with ProcessPoolExecutor(max_workers=args.jobs, max_tasks_per_child=1) as executor:
        for batch_name, device_name in batches:
            qasm_dir = CIRCUITS_DIR / batch_name
            device_path = DEVICES_DIR / f"{device_name}.json"
            _, qcn = QuHardwareInfoReader(device_path).get_hardware_info(chip_type=CHIP_TYPE)
            qubit_chip = qcn.obtain_qubit_n_chip_idx()

            batch = {
                "batch": batch_name,
                "qasm_dir": str(qasm_dir.relative_to(REPO)),
                "device": str(device_path.relative_to(REPO)),
                "results": [],
            }
            payload["batches"].append(batch)

            files = sorted(qasm_dir.glob("*.qasm"))
            if args.circuit:
                files = [f for f in files if any(c in f.name for c in args.circuit)]
            for i, qasm_path in enumerate(files, 1):
                print(f"[{batch_name} on {device_name}] ({i}/{len(files)}) {qasm_path.name}", flush=True)
                result = run_file(qasm_path, device_path, qubit_chip, seeds, args.remote_swap_epr, executor)
                batch["results"].append(result)
                if result["best"]:
                    b = result["best"]
                    print(f"    best seed={b['seed']}  epr={b['epr']}  swaps={b['swaps']}  "
                          f"depth={b['circuit_depth']}", flush=True)
                # Save after every circuit so partial results survive an interruption.
                out.write_text(json.dumps(payload, indent=2))

    out.write_text(json.dumps(payload, indent=2))
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
