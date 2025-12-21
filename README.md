# fed_trainers

Federated learning trainers and sweepers for differentially private SGD (DP‑SGD) and Gradient Embedding Perturbation (GEP) on vision and EMG datasets. The repo includes:
- Training scripts for CIFAR‑10/100, MNIST, and EMG datasets (e.g., putEMG, KeyPressEMG)
- Hyperparameter sweep "sweeper" scripts
- Utilities for dataset preparation, logging, and privacy accounting

Note: Some parts of the repository are research‑oriented and may require dataset preparation steps not yet documented. See TODOs below.

## Stack and Tooling
- Language: Python (pyproject targets Python ^3.10)
- Frameworks/Libraries:
  - PyTorch 2.3, torchvision 0.18
  - scikit‑learn, pandas, numpy, scipy
  - Opacus (differential privacy)
  - backpack‑for‑pytorch
  - Weights & Biases (wandb)
- Package manager: Poetry (preferred) via `pyproject.toml`
- Alternative dependency spec: `requirements.txt` (used by Dockerfile; pins CUDA 12 wheels)
- Optional: Dockerfile (see notes/mismatch below)

## Project Structure
- `fed_trainers/`
  - `datasets/`: dataset utilities (`dataset.py`, `emg_utils.py`, `femnist_utils.py`, `keypressemg_utils.py`)
  - `sweepers/`: hyperparameter sweep entrypoints
    - `dp_sgd/`: e.g., `sweeper_cifar10_dp_no_gp.py`
    - `gep/`: e.g., `sweeper_cifar10_gep_public_no_gp.py`
  - `trainers/`: training logic for DP‑SGD, GEP, and related utilities
    - `dp_sgd/`, `gep/`, `adadpigu/`
    - Common utilities in `trainers/utils.py` and `trainers/model.py`
- `data/`: auxiliary data files (e.g., distance matrices)
- `hyper_sweeper.sh`: convenience shell script to run a sequence of sweepers (expects the scripts in the CWD)
- `pyproject.toml`: Poetry configuration and console scripts
- `requirements.txt`: pinned runtime deps (used by Dockerfile)
- `Dockerfile`: optional container build

## Entry Points and Scripts
Poetry console scripts (defined in `pyproject.toml`):
- `sweeper_cifar10_dp`
  - Runs DP‑SGD sweep for CIFAR‑10: `fed_trainers.sweepers.dp_sgd.sweeper_cifar10_dp_no_gp:main`
- `sweeper_cifar10_gep_public`
  - Runs GEP‑Public sweep for CIFAR‑10: `fed_trainers.sweepers.gep.sweeper_cifar10_gep_public_no_gp:main`

Additional runnable modules (invoke with `python`):
- Trainers (examples):
  - `fed_trainers/trainers/dp_sgd/trainer_cifar10_dp_no_gp.py`
  - `fed_trainers/trainers/gep/trainer_cifar10_gep_public_no_gp.py`
  - Other datasets/variants exist (e.g., `putEMG`, `femnist`) — see corresponding files under `trainers/` and `sweepers/`.
- Sweepers beyond the Poetry scripts can be called directly, e.g.,
  - `fed_trainers/sweepers/dp_sgd/sweeper_cifar10_dp_no_gp.py`
  - `fed_trainers/sweepers/gep/sweeper_cifar10_gep_no_gp.py`

Shell helper:
- `hyper_sweeper.sh` runs a sequence of CIFAR‑10 sweepers with `python3` (intended for a local workspace where those files are present in the working directory).

## Requirements
- Python: 3.10+ recommended (per `pyproject.toml`)
- OS: Linux/macOS recommended
- GPU: Optional; for accelerated training ensure CUDA‑compatible PyTorch is installed
- For wandb logging (optional): a Weights & Biases account and API key

Python dependencies are specified in both `pyproject.toml` (Poetry) and `requirements.txt` (pip). Prefer Poetry for development to avoid version drift.

## Installation
Using Poetry (recommended):
1. Install Poetry: https://python-poetry.org/docs/#installation
2. From repo root run:
   - `poetry env use 3.10` (optional, to select Python 3.10)
   - `poetry install`

Using pip (alternative):
- Create and activate a virtual environment with Python 3.10+
- Install dependencies:
  - `pip install --upgrade pip`
  - `pip install -r requirements.txt`

Note: The `requirements.txt` pins CUDA 12 wheels (`nvidia‑*`, `torch`, `torchvision`). Ensure your system’s CUDA/toolkit drivers match these versions, or install CPU wheels as appropriate.

## Running
Examples below use Poetry. You can drop `poetry run` if you activated a venv and used pip.

- Run CIFAR‑10 DP‑SGD sweeps (Poetry script):
  - `poetry run sweeper_cifar10_dp`

- Run CIFAR‑10 GEP‑Public sweeps (Poetry script):
  - `poetry run sweeper_cifar10_gep_public`

- Run a single trainer directly (CIFAR‑10 DP‑SGD):
  - `poetry run python fed_trainers/trainers/dp_sgd/trainer_cifar10_dp_no_gp.py \
    --data-name cifar10 \
    --num-clients 500 \
    --classes-per-client 2 \
    --num-client-agg 5 \
    --num-steps 20 \
    --batch-size 64 \
    --optimizer sgd \
    --lr 0.01 \
    --clip 1.0 \
    --noise-multiplier 0.1 \
    --wandb false`

Common flags (may vary by trainer):
- `--data-name`: one of `cifar10`, `cifar100`, `putEMG`, `mnist` (others like `femnist` appear in some modules)
- `--data-path`: dataset directory (default `data/`)
- `--num-clients`, `--num-client-agg`, `--classes-per-client`
- Optimization: `--optimizer`, `--lr`, `--wd`, `--batch-size`, `--inner-steps`
- Privacy: `--clip`, `--noise-multiplier`
- Logging/Eval: `--eval-every`, `--eval-after`, `--log-dir`, `--csv-path`, `--wandb {true|false}`

Sweeper specifics:
- DP‑SGD sweeper (`sweeper_cifar10_dp_no_gp.py`) constructs runs by spawning the corresponding trainer (e.g., CIFAR‑10) with grids over `sigma` (noise multiplier), `lr`, `clip`, `seed`, etc.
- GEP‑Public sweeper (`sweeper_cifar10_gep_public_no_gp.py`) defines a Weights & Biases sweep configuration and calls the training function programmatically.

## Datasets
- CIFAR‑10/100 and MNIST are typically auto‑downloaded by torchvision under the specified `--data-path`.
- EMG datasets (`putEMG`, KeyPressEMG) and FEMNIST likely require manual download/preparation.
  - TODO: Add dataset download links and preparation instructions for EMG/FEMNIST datasets.

## Environment Variables
- `WANDB_API_KEY`: required if enabling `--wandb true` for Weights & Biases logging.
- `CUDA_VISIBLE_DEVICES`: select GPU(s) to use, or use trainer flags `--gpu`/`--gpus` where available.

## Docker (optional)
A `Dockerfile` is provided that:
- Uses `python:3.8-slim`
- Installs dependencies from `requirements.txt`
- Copies selected trainer/sweeper files into `/workspace`
- Default command runs `trainer_putEMG_gep_no_gp.py`

Notes/TODOs:
- Python version mismatch: `pyproject.toml` targets Python ^3.10, but Dockerfile uses Python 3.8. Consider updating the base image to Python 3.10.
- Dependency source mismatch: project prefers Poetry; Dockerfile installs via `requirements.txt`. Consider aligning the container build with Poetry or regenerating `requirements.txt` from Poetry.
- The Dockerfile currently copies only selected files; consider switching to an installable package copy (`pip install .`) or copying the full project to ensure all imports resolve.

## Testing
- No test suite is present in the repository.
- TODO: Add unit tests for dataset loaders, training loops, and privacy accounting. Integrate with `pytest` and CI.

## Development Tips
- Code style: follow existing formatting patterns in each module.
- Reproducibility: use `--seed` and see `trainers/utils.py:set_seed`.
- Logging: trainers set up file and console logging via `set_logger` with `--log-dir`, `--log-name`, and `--log-level`.

## License
- No license file is included.
- TODO: Add an open‑source license (e.g., MIT, Apache‑2.0) or specify the proprietary terms.

## Acknowledgments
- Built on PyTorch and related libraries; includes privacy utilities (Opacus) and optional experiment tracking via Weights & Biases.
