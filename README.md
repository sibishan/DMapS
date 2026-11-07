## Running the experiment for BOLTSABRE

The experiment in `experiments/` maps and routes circuits with DMapS (CPA mapping + CAR routing) on devices A–F:

- **Original circuits** (`experiments/data/in/circuits/`): each circuit declares only the qubits it uses.
- **Padded circuits** (`experiments/data/in/circuits_alt/`): the same circuits with extra idle qubits declared.
  DMapS cannot run these, so they are switched off (see [Padded circuits](#padded-circuits) below).

Each circuit group is routed on one device:

| Group | Device | Cores |
|---|---|---|
| `qasm_25` | A | four 4×4 cores, ring |
| `qasm_64` | B | four 5×5 cores, ring |
| `qasm_80` | C | four 6×5 cores, ring |
| `qasm_95` | D | six 5×5 cores, 2×3 core grid |
| `qasm_200` | E | eight 7×5 cores, 2×4 core grid |
| `qasm_360` | F | twelve 7×6 cores, 3×4 core grid |

The devices are defined in `experiments/devices/*.json`.

```bash
pip install -e .
python experiments/run.py

# Only some batches / circuits (substring match), 3 seeds, 4 parallel jobs
python experiments/run.py --batch qasm_25 --circuit qft --circuit ghz --seeds 3 -j 4
```

It writes `experiments/data/out/dmaps_results.json`, overwriting any earlier run. For each circuit the file lists every seed's
metrics and validity check, plus the best seed. A run is valid if every two-qubit gate sits on a device coupling.

| Option | Default | Description |
|---|---|---|
| `--seeds N` | 5 | Seeds per circuit |
| `--start-seed S` | 0 | First seed |
| `--out-dir DIR` | `experiments/data/out` | Output directory |
| `--batch LABEL` | all | Run only this group (e.g. `qasm_25`); repeatable |
| `--circuit NAME` | all | Run only circuits whose file name contains `NAME`; repeatable |
| `-j`, `--jobs` | CPU count | Seeds run in parallel |

To change what runs, edit the top of `experiments/run.py`:

- `BATCHES`: which circuit group runs on which device (comment out a row to skip it)
- `CIRCUIT_SETS`: which circuit sets run, in what order (the output-file suffix comes from the set name)
- `REMOTE_SWAP_COST`, `TELEGATE_COST`: EPR pairs charged per remote SWAP and per telegate

### Metrics

The metrics follow the DMapS paper (Sec. IV) and are counted from the routed circuit:

- **Remote SWAP** (`remote_swaps`): SWAP gates between qubits on different chips
- **Telegate** (`telegates`): CX gates between qubits on different chips (remote CNOTs)
- **Inter-core** (`inter_core`) = remote SWAP + telegate
- **EPR cost** (`epr_cost`) = 2 × remote SWAP + 1 × telegate
- **SWAP** (`swaps`): intra-chip SWAPs

The best seed is the valid run with the lowest EPR cost, then the fewest SWAPs, then the lowest seed. A seed always gives the same result.

### Padded circuits

The padded circuits have the same gates as the originals but declare qubits that no gate touches
(e.g. `qreg q[41]` with 25 used). DMapS never counts or places qubits without gates, so its partitioner fails on the
first idle qubit (`src/partitioner/qucirc_partitioner.py:104`, `KeyError`). Supporting them would mean changing DMapS
itself, so `circuits_alt` is commented out of `CIRCUIT_SETS`. If both sets are enabled, `run.py` also writes
`dmaps_results_alt.json` and a side-by-side `dmaps_comparison.json`.

### Results and analysis

`experiments/data/out/dmaps_results.json` holds the results for the original circuits.

`experiments/data/out/analysis.ipynb` breaks the inter-core count down per circuit, in the layout of the paper's Table III:
remote SWAPs, telegates, inter-core total, EPR cost and SWAPs, with geometric means per group and over the whole suite.
It ends with a note on why there is no padded comparison.

---

# DMapS
**DMapS** is **end-to-end quantum compiler** designed for **near-term distributed quantum computing architectures**, integrating qubit mapping and routing algorithms. These algorithms improve the execution efficiency of compiled quantum programs by optimizing EPR pair overhead and local SWAP gate insertions.

## 📦 Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/RoccoLoter/DMapS.git
cd DMapS
pip install -e .
```

## 🛠️ Usage
More usage tutorials can be found [here](tests/usage.ipynb).​

## 📂 Project Structure
```bash
DMapS/
├── src/                            # Core implementation
│   ├── comparion_algorithms/       # Core compiler logic
│   └── frontend/                   # Config processing implementation​
│   └── mapper/                     # Qubit mapping implementation
│   └── partitioner/                # Circuit partitioning implementation 
│   └── router/                     # Qubit routing implementation
│   └── global_config.py            # Relative path definition
├── tests/                          # Experimental files and usage tutorials​
│   ├── benchmark/                  # OpenQASM benchmark files
│   └── hardware_info/              # Reconfiguration files for quantum chip networks​
│   └── usage.ipynb                 # Usage tutorials​
```

## 📄 License
This project is licensed under the Apache 2.0.

## 📬 Contact 
- The code comments are still insufficient, and we will work on improving them.
- For questions, please contact listenwetnessluo@163.com or open an issue on GitHub.

## ✨ How to Cite the DMapS Paper
If you consider citing the DMapS paper:
```bash
@article{luo2025dmaps,
  title={DMapS: End-to-end Qubit Mapping and Routing for Distributed Quantum Computing Architectures},
  author={Luo, Tingyu and Zheng, Yuzhen and Deng, Yuxin and Fu, Xiang},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2025},
  publisher={IEEE}
}
```
