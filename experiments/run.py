#!/usr/bin/env python3
"""Compare the "circuits" and "circuits_alt" circuit sets on the base (A-F) devices.

Every circuit batch of both sets (read from experiments/data/in/) is run through
DMapS (CPA mapping + CAR routing) for a range of seeds, and the results are
collected into:

  <out_dir>/dmaps_results.json       (circuits     on A-F)
  <out_dir>/dmaps_results_alt.json   (circuits_alt on A-F)

At the end the circuits of both sets are paired by (batch, file name); the
pairing is written to <out_dir>/dmaps_comparison.json and printed as a table of
their best results side by side plus alt - base deltas.

Metrics (DMapS paper, Sec. IV):
  telegates    = CX gates between qubits on different chips (remote CNOT)
  remote_swaps = SWAP gates between qubits on different chips
  inter_core   = remote_swaps + telegates
  epr_cost     = remote_swaps * REMOTE_SWAP_COST + telegates * TELEGATE_COST

The best run of each circuit is the valid run with the lowest epr_cost, ties
broken by fewest local swaps, then lowest seed.

Usage:
  experiments/run.py [--seeds N] [--start-seed S] [--out-dir DIR] [--batch qasm_25 ...]
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
from concurrent.futures import ProcessPoolExecutor, as_completed
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

IN_DIR = REPO / "experiments" / "data" / "in"
DEVICES_DIR = REPO / "experiments" / "devices"
OUT_DIR = REPO / "experiments" / "data" / "out"
NUM_SEEDS = 5
START_SEED = 0
JOBS = os.cpu_count()
# EPR pairs per remote operation, as in the DMapS paper's metric.
REMOTE_SWAP_COST = 2
TELEGATE_COST = 1
CHIP_TYPE = "ibm"
INTRA_CHIP_ALL2ALL = False
# Circuit sets to compare (subdirectories of IN_DIR). The first is the baseline.
# "circuits_alt" is disabled for now: DMapS crashes on its padded (idle-qubit) registers.
CIRCUIT_SETS = ["circuits"]  # , "circuits_alt"]
# (label, device) -- each label is a qasm_* subdirectory of every circuit set.
BATCHES = [
    ("qasm_25", "A"),
    ("qasm_64", "B"),
    ("qasm_80", "C"),
    ("qasm_95", "D"),
    ("qasm_200", "E"),
    ("qasm_360", "F"),
]


def out_suffix(set_name):
    """Output file suffix for a circuit set: dmaps_results<suffix>.json."""
    if set_name == "circuits":
        return ""
    if set_name.startswith("circuits_"):
        return "_" + set_name[len("circuits_"):]
    return "_" + set_name

ScChip.__hash__ = lambda self: hash(self._chip_idx)

class _SerialPool:
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
    def wrapper(*args, **kwargs):
        return sorted(obtain_swaps(*args, **kwargs), key=lambda s: (_bit_key(s[0]), _bit_key(s[1])))
    return wrapper


_multi_mode_swap_init = mapper.local_sabre_mapper.MultiModeSwap.__init__


def _seeded_multi_mode_swap_init(self, *args, **kwargs):
    _multi_mode_swap_init(self, *args, **kwargs)
    if self.seed is None:
        self.seed = np.random.randint(0, np.iinfo(np.int32).max)


mapper.local_sabre_mapper.MultiModeSwap.__init__ = _seeded_multi_mode_swap_init
mapper.local_sabre_mapper.MultiModeSwap._obtain_swaps = _sorted_swaps(
    mapper.local_sabre_mapper.MultiModeSwap._obtain_swaps)
router.car_layout_routing.CARSwap._obtain_swaps = _sorted_swaps(
    router.car_layout_routing.CARSwap._obtain_swaps)

_kahypar_partition = kahypar.partition
_kahypar_seed = None


def _seeded_partition(hypergraph, context):
    context.setSeed(_kahypar_seed)
    return _kahypar_partition(hypergraph, context)


kahypar.partition = _seeded_partition


def bar(done, total, width=24):
    filled = int(width * done / total) if total else width
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def loader(prefix, done, total, extra=""):
    sys.stdout.write(f"\r\033[K  {prefix} seeds {bar(done, total, 10)} {extra}")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


@contextlib.contextmanager
def silenced():
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


def single_register_qasm(qasm_path: Path, tmp_dir: str) -> Path:
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    if len(circuit.qregs) == 1:
        return qasm_path
    flat = QuantumCircuit(QuantumRegister(circuit.num_qubits, "q"), *circuit.cregs,
                          name=circuit.name, global_phase=circuit.global_phase)
    flat.compose(circuit, qubits=flat.qubits, clbits=flat.clbits, inplace=True)
    out = Path(tmp_dir) / qasm_path.name
    flat.qasm(filename=str(out))
    return out


def load_device(device_name):
    device_path = DEVICES_DIR / f"{device_name}.json"
    _, qcn = QuHardwareInfoReader(device_path).get_hardware_info(
        chip_type=CHIP_TYPE, intra_chip_all2all=INTRA_CHIP_ALL2ALL)
    edges = frozenset(frozenset(e) for e in qcn.obtain_total_chip_network().edges())
    return device_path, qcn.obtain_qubit_n_chip_idx(), edges


def analyse(circuit: QuantumCircuit, qubit_chip: dict, edges: frozenset):
    counts = {"remote_cx": 0, "remote_swaps": 0, "swaps": 0}
    errors = []
    for instr in circuit.data:
        if len(instr.qubits) != 2:
            continue
        name = instr.operation.name
        q0, q1 = (circuit.find_bit(q).index for q in instr.qubits)
        if frozenset((q0, q1)) not in edges:
            errors.append(f"{name} on uncoupled qubits ({q0}, {q1})")
        remote = qubit_chip[q0] != qubit_chip[q1]
        if name == "cx" and remote:
            counts["remote_cx"] += 1
        elif name == "swap":
            counts["remote_swaps" if remote else "swaps"] += 1
    return counts, errors


def run_one(qasm_path: Path, device_path: Path, qubit_chip: dict, edges: frozenset, seed: int) -> dict:
    global _kahypar_seed
    random.seed(seed)
    np.random.seed(seed)
    _kahypar_seed = seed

    t0 = time.perf_counter()
    try:
        compiler = MultiModeRouting()
        with tempfile.TemporaryDirectory() as tmp_dir, silenced():
            _, _, routed = compiler.cpa_map_car_rout(
                origin_qasm_fn=single_register_qasm(qasm_path, tmp_dir),
                chip_info_fn=device_path,
                chip_type=CHIP_TYPE,
                intra_chip_all2all=INTRA_CHIP_ALL2ALL,
            )
    except Exception:
        exc = traceback.format_exc(limit=3)
        return {
            'seed': seed, 'layout': 'cpa',
            'swaps': None, 'remote_swaps': None, 'telegates': None,
            'inter_core': None, 'epr_cost': None,
            'circuit_depth': None,
            'solved_deadlocks': None, 'solving_deadlock': False,
            'num_operations': 0, 'valid': False,
            'errors': [f"exception: {exc.strip().splitlines()[-1]}"],
            'mapping_time_s': None, 'routing_time_s': None,
            'runtime_s': time.perf_counter() - t0,
        }
    runtime_s = time.perf_counter() - t0

    counts, errors = analyse(routed, qubit_chip, edges)
    return {
        'seed': seed,
        'layout': 'cpa',
        'swaps': counts['swaps'],
        'remote_swaps': counts['remote_swaps'],
        'telegates': counts['remote_cx'],
        'inter_core': counts['remote_swaps'] + counts['remote_cx'],
        'epr_cost': REMOTE_SWAP_COST * counts['remote_swaps'] + TELEGATE_COST * counts['remote_cx'],
        'circuit_depth': routed.depth(),
        'solved_deadlocks': None,
        'solving_deadlock': False,
        'num_operations': len(routed.data),
        'valid': not errors,
        'errors': errors[:10] + ([f"... {len(errors) - 10} more"] if len(errors) > 10 else []),
        'mapping_time_s': compiler.mapping_time,
        'routing_time_s': compiler.routing_time,
        'runtime_s': runtime_s,
    }


def run_file(qasm_path, device_name, device, executor, seeds, progress=""):
    device_path, qubit_chip, edges = device
    name = qasm_path.name
    tag = f"{progress} {name}" if progress else name
    num_seeds = len(seeds)

    loader(tag, 0, num_seeds, extra="running...")
    futures = [executor.submit(run_one, qasm_path, device_path, qubit_chip, edges, seed)
               for seed in seeds]
    runs = []
    for future in as_completed(futures):
        r = future.result()
        runs.append(r)
        if len(runs) < num_seeds:
            loader(tag, len(runs), num_seeds, extra=f"seed {r['seed']} took {r['runtime_s']:.1f}s")
    runs.sort(key=lambda r: r['seed'])
    loader(tag, num_seeds, num_seeds, extra=f"done ({sum(r['runtime_s'] for r in runs):.1f}s of runs)")

    valid_runs = [r for r in runs if not r['solving_deadlock'] and r['valid']]
    best = None
    if valid_runs:
        best = min(valid_runs, key=lambda r: (r['epr_cost'], r['swaps'], r['seed']))

    return {
        'file': name,
        'path': str(qasm_path.relative_to(REPO)),
        'arch': device_name,
        'device': str(device_path.relative_to(REPO)),
        'num_seeds': num_seeds,
        'best_seed': best['seed'] if best else None,
        'best': {
            'seed': best['seed'],
            'layout': best['layout'],
            'remote_swaps': best['remote_swaps'],
            'telegates': best['telegates'],
            'inter_core': best['inter_core'],
            'epr_cost': best['epr_cost'],
            'swaps': best['swaps'],
            'circuit_depth': best['circuit_depth'],
            'mapping_time_s': best['mapping_time_s'],
            'routing_time_s': best['routing_time_s'],
            'valid': best['valid'],
            'errors': best['errors'],
            'runtime_s': best['runtime_s'],
        } if best else None,
        'num_valid_runs': len(valid_runs),
        'all_valid': all(r['valid'] for r in runs),
        'sabre_fallbacks': 0,   # DMapS has no layout fallback; kept for parity with boltsabre
        'runs': runs,
    }


def qasm_files(qasm_dir, circuit_filter):
    files = sorted(Path(qasm_dir).glob("*.qasm"))
    if circuit_filter:
        files = [f for f in files if any(c in f.name for c in circuit_filter)]
    return files


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def run_set(set_name, batches_to_run, circuit_filter, seeds, out_dir, pass_no, counter, executor):
    suffix = out_suffix(set_name)
    print(f"===== Pass {pass_no}/{len(CIRCUIT_SETS)}: dmaps_results{suffix} "
          f"({set_name} on {'-'.join([batches_to_run[0][1], batches_to_run[-1][1]])}) =====\n")
    out_path = out_dir / f"dmaps_results{suffix}.json"
    payload = {
        'generated_at': datetime.now().isoformat(),
        'method': "DMapS cpa_map_car_rout",
        'variant': set_name,
        'num_seeds': len(seeds),
        'start_seed': seeds[0],
        'layout_protocol': (f"DMapS CPA mapping + CAR routing on seeds {seeds[0]}..{seeds[-1]}; "
                            f"best by (epr_cost, swaps)"),
        'config': {
            'chip_type': CHIP_TYPE,
            'intra_chip_all2all': INTRA_CHIP_ALL2ALL,
            'remote_swap_cost': REMOTE_SWAP_COST,
            'telegate_cost': TELEGATE_COST,
            'jobs': executor._max_workers,
        },
        'batches': [],
    }

    for label, device_name in batches_to_run:
        qasm_dir = IN_DIR / set_name / label
        files = qasm_files(qasm_dir, circuit_filter)
        print(f"[{set_name}/{label}] {device_name}: {len(files)} qasm files in {rel(qasm_dir)}\n")
        device = load_device(device_name)

        batch = {
            'batch': label,
            'qasm_dir': rel(qasm_dir),
            'arch': device_name,
            'device': rel(device[0]),
            'results': [],
        }
        payload['batches'].append(batch)

        for path in files:
            counter['done'] += 1
            progress = f"overall {bar(counter['done'], counter['total'])} |"
            result = run_file(path, device_name, device, executor, seeds, progress)
            batch['results'].append(result)
            if result['best']:
                b = result['best']
                print(f"    best seed={b['seed']}  epr_cost={b['epr_cost']}  inter-core={b['inter_core']} "
                      f"(remote_swaps={b['remote_swaps']}, telegates={b['telegates']})  "
                      f"swaps={b['swaps']}  depth={b['circuit_depth']}  valid={b['valid']}")
            else:
                print("    no valid run (all failed validation or raised)")
            for r in result['runs']:
                if r['errors'] and r['errors'][0].startswith("exception"):
                    print(f"    seed {r['seed']} FAILED: {r['errors'][0]}")
                elif not r['valid']:
                    print(f"    seed {r['seed']} INVALID: {r['errors'][:3]}")
            out_path.write_text(json.dumps(payload, indent=2))
        print()

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote results to {rel(out_path)}\n")
    return payload


def _pick(best):
    if best is None:
        return None
    keys = ("seed", "remote_swaps", "telegates", "inter_core", "epr_cost", "swaps",
            "circuit_depth", "runtime_s")
    return {k: best.get(k) for k in keys}


def _delta(base, alt, key):
    if base is None or alt is None or base.get(key) is None or alt.get(key) is None:
        return None
    return alt[key] - base[key]


def compare(base_name, base, alt_name, alt, out_dir):
    """Pair circuits by (batch, file) across the two sets; write JSON and print a table."""
    def index(payload):
        return {(b['batch'], r['file']): {**r, 'batch': b['batch']}
                for b in payload['batches'] for r in b['results']}

    B, A = index(base), index(alt)
    order = {label: i for i, (label, _) in enumerate(BATCHES)}
    keys = sorted(set(B) | set(A), key=lambda k: (order.get(k[0], len(order)), k[0], k[1]))
    circuits = []
    for key in keys:
        b, a = B.get(key), A.get(key)
        any_ = b or a
        bb, ab = (b or {}).get('best'), (a or {}).get('best')
        circuits.append({
            'batch': any_['batch'], 'file': any_['file'], 'arch': any_['arch'],
            base_name: _pick(bb),
            alt_name: _pick(ab),
            'delta_alt_minus_base': {k: _delta(bb, ab, k) for k in
                                     ("epr_cost", "inter_core", "remote_swaps", "telegates", "swaps", "circuit_depth")},
        })

    cmp_path = out_dir / "dmaps_comparison.json"
    cmp_path.write_text(json.dumps({
        'generated_at': datetime.now().isoformat(),
        'base': base_name, 'alt': alt_name,
        'config': base['config'],
        'num_seeds': base['num_seeds'], 'start_seed': base['start_seed'],
        'circuits': circuits,
    }, indent=2))

    def s(v):
        return "-" if v is None else str(v)

    header = ["batch", "arch", "file", "epr_base", "epr_alt", "d_epr", "ic_base", "ic_alt", "d_ic", "sw_base", "sw_alt", "d_sw",
              "dep_base", "dep_alt", "d_dep"]
    rows = [header]
    for c in circuits:
        cb, ca, d = c[base_name] or {}, c[alt_name] or {}, c['delta_alt_minus_base']
        rows.append([c['batch'], c['arch'], c['file'].replace("_nativegates_ibm_qiskit_opt3", ""),
                     s(cb.get('epr_cost')), s(ca.get('epr_cost')), s(d['epr_cost']),
                     s(cb.get('inter_core')), s(ca.get('inter_core')), s(d['inter_core']),
                     s(cb.get('swaps')), s(ca.get('swaps')), s(d['swaps']),
                     s(cb.get('circuit_depth')), s(ca.get('circuit_depth')), s(d['circuit_depth'])])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]

    print(f"===== Comparison: {base_name} vs {alt_name} (best runs, delta = alt - base) =====")
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip())
    print(f"\nWrote comparison to {rel(cmp_path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=NUM_SEEDS,
                        help=f"number of seeds per circuit (default: {NUM_SEEDS})")
    parser.add_argument("--start-seed", type=int, default=START_SEED,
                        help=f"first seed (default: {START_SEED})")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"output directory (default: {rel(OUT_DIR)})")
    parser.add_argument("--batch", action="append",
                        help="only run this batch label, e.g. qasm_25 (repeatable)")
    parser.add_argument("--circuit", action="append",
                        help="only run circuits whose file name contains this (repeatable)")
    parser.add_argument("-j", "--jobs", type=int, default=JOBS,
                        help="seeds run in parallel (default: number of CPUs)")
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    batches_to_run = [b for b in BATCHES if not args.batch or b[0] in args.batch]
    if not batches_to_run:
        parser.error("--batch matched no batch")
    for set_name in CIRCUIT_SETS:
        if not (IN_DIR / set_name).is_dir():
            parser.error(f"{rel(IN_DIR / set_name)} not found")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    total = sum(len(qasm_files(IN_DIR / set_name / label, args.circuit))
                for set_name in CIRCUIT_SETS for label, _ in batches_to_run)
    counter = {'done': 0, 'total': total}

    payloads = []
    with ProcessPoolExecutor(max_workers=args.jobs, max_tasks_per_child=1) as executor:
        for pass_no, set_name in enumerate(CIRCUIT_SETS, 1):
            payloads.append(run_set(set_name, batches_to_run, args.circuit, seeds, out_dir,
                                    pass_no, counter, executor))

    if len(CIRCUIT_SETS) >= 2:
        compare(CIRCUIT_SETS[0], payloads[0], CIRCUIT_SETS[1], payloads[1], out_dir)


if __name__ == "__main__":
    main()
