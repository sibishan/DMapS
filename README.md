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

## 🧪 Running Experiments
`scripts/run_experiment.py` runs DMapS (CPA mapping + CAR routing) on every circuit in `experiments/circuits/<batch>/` against a device in `experiments/devices/`, once per seed, and writes all runs plus the best run (fewest EPR pairs) to one JSON file. DMapS needs Python 3.11.

```bash
# Default: qasm_25 on G, qasm_64 on H, 10 seeds, all CPU cores
python scripts/run_experiment.py -o experiments/results/results.json

# Alternative devices
python scripts/run_experiment.py --batch qasm_25:G_alt --batch qasm_64:H_alt \
    -o experiments/results/results_alt.json

# Only some circuits (substring match), 5 seeds, 4 parallel jobs
python scripts/run_experiment.py --batch qasm_25:G --circuit qft --circuit ghz \
    --num-seeds 5 -j 4
```

| Option | Default | Description |
|---|---|---|
| `--num-seeds` | 10 | Seeds per circuit |
| `--start-seed` | 0 | First seed |
| `--batch DIR:DEVICE` | `qasm_25:G`, `qasm_64:H` | Circuit folder and device name; repeatable |
| `--circuit NAME` | all | Run only circuits whose filename contains `NAME`; repeatable |
| `-j`, `--jobs` | CPU count | Seeds run in parallel |
| `--remote-swap-epr` | 2 | EPR pairs per remote SWAP |
| `-o`, `--output` | `experiments/results/dmaps_<timestamp>.json` | Output file |

EPR count follows the DMapS paper: a remote CNOT costs 1 EPR pair and a remote SWAP costs 2. A seed always gives the same result. Devices are converted from telesabre format with `scripts/telesabre_device_to_dmaps.py`.

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
