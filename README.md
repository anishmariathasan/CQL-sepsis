# Conservative Q-Learning for Offline Sepsis Treatment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements **Conservative Q-Learning (CQL)** for offline reinforcement learning applied to sepsis treatment in the ICU. Using the ICU-Sepsis benchmark environment, I investigate whether offline RL can learn effective treatment policies from historical clinician data without requiring risky online environment interaction.

### Key Findings

- **CQL (α=0.5)** achieves **86.0%** survival rate, outperforming DQN (82.0%) and random baseline (81.7%)
- **DQN without conservatism** exhibits training instability (loss explosion) due to Q-value overestimation for out-of-distribution actions
- **Behaviour Cloning** provides a strong baseline (84.0%) by imitating clinician actions
- The conservatism parameter α is crucial: too low leads to OOD overestimation, too high leads to underfitting

### Objectives

1. **Safe Offline Learning**: Train treatment policies from historical clinician data without risky environment interaction
2. **Conservative Q-Function**: Prevent overestimation of Q-values for out-of-distribution actions
3. **Clinical Applicability**: Learn actionable treatment recommendations for vasopressors and IV fluids
4. **Comprehensive Evaluation**: Alpha sensitivity analysis, algorithm comparison, and effect size analysis

## Repository Structure

```
CQL-sepsis/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment file
├── pyproject.toml                     # Package configuration
├── LICENSE                            # MIT License
│
├── src/                               # Source code
│   ├── algorithms/
│   │   ├── cql.py                    # Conservative Q-Learning
│   │   ├── bc.py                     # Behaviour Cloning baseline
│   │   └── dqn.py                    # DQN baseline
│   ├── environments/
│   │   └── icu_sepsis_wrapper.py     # ICU-Sepsis environment wrapper
│   ├── data/
│   │   ├── replay_buffer.py          # Offline dataset management
│   │   └── data_collection.py        # Expert policy data collection
│   └── utils/
│       ├── logging.py                # Experiment logging
│       ├── evaluation.py             # Policy evaluation
│       └── plotting.py               # Visualisation utilities
│
├── configs/                           # YAML configuration files
│   ├── cql_default.yaml              # Default CQL hyperparameters
│   └── experiment_grid.yaml          # Alpha sweep configuration
│
├── scripts/                           # Executable scripts
│   ├── 01_install_environment.py     # Verify installation
│   ├── 02_collect_offline_data.py    # Collect expert policy data
│   ├── 03_train_cql.py               # Train CQL with alpha sweep
│   ├── 04_train_baselines.py         # Train BC and DQN baselines
│   ├── 05_evaluate_policies.py       # Evaluate trained policies
│   ├── reproduce_all.ps1             # Windows reproduction script
│   └── reproduce_all.sh              # macOS/Linux reproduction script
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_environment_exploration.ipynb  # Environment analysis
│   └── 02_comprehensive_evaluation.ipynb # Results & visualisations
│
├── data/                              # Generated data
│   └── behavior_policy.pkl           # Expert policy offline dataset
│
├── results/                           # Experimental results
│   ├── cql_alpha_*_seed_*/           # CQL results by alpha and seed
│   ├── bc_seed_*/                    # Behaviour Cloning results
│   ├── dqn_seed_*/                   # DQN results
│   └── comprehensive_results.csv     # Summary statistics
│
├── figures/                           # Generated figures (PNG)
│
└── tests/                             # Unit tests
    └── test_cql.py
```

## Quick Start

### Prerequisites

- Python 3.10+ (recommended)
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration on Windows/Linux)

### Installation

#### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/anishmariathasan/CQL-sepsis.git
cd CQL-sepsis

# Create conda environment from file
conda env create -f environment.yml
conda activate cql-sepsis

# Verify installation
python scripts/01_install_environment.py
```

#### Option 2: pip

```bash
# Clone repository
git clone https://github.com/anishmariathasan/CQL-sepsis.git
cd CQL-sepsis

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/01_install_environment.py
```

> **⚠️ macOS / Non-CUDA Systems**: The `environment.yml` includes `cudatoolkit=11.8` which will fail on systems without CUDA support. Before running `conda env create`, edit `environment.yml` and remove the line:
> ```yaml
> - cudatoolkit=11.8
> ```
> PyTorch will automatically use CPU-only mode.

### Reproduce All Experiments

To run the complete experimental pipeline (data collection → training → evaluation):

#### Windows (PowerShell)

```powershell
.\scripts\reproduce_all.ps1
```

#### macOS / Linux (Bash)

```bash
bash scripts/reproduce_all.sh
```

The full pipeline took approximately **4-5 hours** on my machine (i7-11370H, NVIDIA RTX 3050 Ti Laptop GPU) and will:
1. Collect offline data using the ICU-Sepsis expert policy (~50,000 transitions)
2. Train CQL with α ∈ {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0} across 3 seeds
3. Train BC and DQN baselines across 3 seeds
4. Evaluate all trained policies

Then run the `02_comprehensive_evaluation` notebook to get figures saved to  `figures/`, and summary statistics to `results/comprehensive_results.csv`.

## Experiments

### 1. CQL Alpha Sensitivity Analysis

Sweep over the conservatism coefficient α ∈ {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0} to analyse the trade-off between conservatism and optimality.

| Alpha | Survival Rate | Std Dev |
|-------|--------------|---------|
| 0.0   | 82.0%        | 2.8%    |
| 0.1   | 85.7%        | 4.7%    |
| **0.5** | **86.0%**  | **3.6%** |
| 1.0   | 84.7%        | 2.9%    |
| 2.0   | 85.3%        | 2.5%    |
| 5.0   | 83.3%        | 1.7%    |
| 10.0  | 84.0%        | 2.9%    |

### 2. Algorithm Comparison

| Algorithm | Survival Rate | Std Dev |
|-----------|--------------|---------|
| **CQL (α=0.5)** | **86.0%** | 3.6% |
| Behaviour Cloning | 84.0% | 2.9% |
| DQN | 82.0% | 2.8% |
| Random | 81.7% | 3.7% |

### 3. Key Observations

- **DQN Loss Explosion**: Without conservatism, DQN's TD loss explodes during training due to Q-value overestimation for out-of-distribution actions. This validates the need for CQL's conservative penalty.
- **Optimal α**: α=0.5 achieves the best balance—low α behaves like DQN (unstable), high α is overly conservative.
- **Expert Policy Coverage**: The offline data is collected using the ICU-Sepsis benchmark's built-in expert policy, which covers ~51.7% of the state-action space (realistic clinical coverage).

## Algorithm Details

### Conservative Q-Learning (CQL)

CQL adds a conservative regulariser to standard Q-learning to prevent overestimation of Q-values for out-of-distribution actions:

$$\mathcal{L}_{\text{CQL}}(\theta) = \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q_\theta(s, a)) - \mathbb{E}_{a \sim \pi_\beta}[Q_\theta(s, a)] \right] + \frac{1}{2} \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (Q_\theta(s,a) - \mathcal{B}^\pi \hat{Q}_{\bar{\theta}}(s,a))^2 \right]$$

**Key Components:**
- **Conservative Penalty** (first term): Pushes down Q-values for all actions while pushing up Q-values for actions seen in the dataset
- **TD Loss** (second term): Standard temporal difference loss with Double DQN
- **Alpha (α)**: Controls the degree of conservatism (higher = more conservative)

### ICU-Sepsis Environment

The ICU-Sepsis benchmark provides a realistic MDP for sepsis treatment:

**Action Space**: 25 discrete actions (5×5 grid)
- **Vasopressors**: 5 levels (none, low, medium, high, very high)
- **IV Fluids**: 5 levels (none, low, medium, high, very high)

**State Space**: 716 discrete states representing patient physiological conditions
- Vital signs (heart rate, blood pressure, temperature)
- Lab values (lactate, creatinine, glucose)
- Demographics and comorbidities

**Expert Policy**: Built-in clinician policy derived from real ICU data (~51.7% state-action coverage)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_environment_exploration.ipynb` | Explore the ICU-Sepsis environment, action space, and expert policy |
| `02_comprehensive_evaluation.ipynb` | Training curves, alpha sweep, algorithm comparison, and effect size analysis |

## References

1. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). **Conservative Q-Learning for Offline Reinforcement Learning**. *NeurIPS*. — The original CQL algorithm implemented in this project.
2. Choudhary, S., et al. (2024). **ICU-Sepsis: A Benchmark MDP Built from Real Medical Data**. *RLC*. — The benchmark environment used for evaluation.
3. Komorowski, M., et al. (2018). **The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care**. *Nature Medicine*. — Foundational work on RL for sepsis treatment that motivates this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This is a university research project for educational purposes. The learned policies should not be used for actual medical decision-making without proper clinical validation and regulatory approval.
