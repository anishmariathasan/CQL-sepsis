# Conservative Q-Learning for Offline Sepsis Treatment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements **Conservative Q-Learning (CQL)** for offline reinforcement learning applied to sepsis treatment in the ICU. Using the ICU-Sepsis benchmark environment, we investigate whether offline RL can learn effective treatment policies from historical data without requiring risky online environment interaction.

### Objectives

1. **Safe Offline Learning**: Train treatment policies without risky environment interaction
2. **Conservative Q-Function**: Avoid overestimation of Q-values for out-of-distribution actions
3. **Clinical Applicability**: Learn actionable treatment recommendations for vasopressors and IV fluids
4. **Comprehensive Evaluation**: Safety analysis, policy interpretation, and ablation studies

## Repository Structure

```
CQL-sepsis/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package configuration
├── LICENSE                            # MIT License
│
├── src/                               # Source code
│   ├── algorithms/
│   │   ├── cql.py                    # Conservative Q-Learning
│   │   ├── bc.py                     # Behaviour Cloning baseline
│   │   └── dqn.py                    # DQN baseline
│   ├── environments/
│   │   └── icu_sepsis_wrapper.py     # Environment wrapper
│   ├── data/
│   │   ├── replay_buffer.py          # Offline dataset management
│   │   └── data_collection.py        # Data collection utilities
│   └── utils/
│       ├── logging.py                # Experiment logging
│       ├── evaluation.py             # Policy evaluation
│       └── plotting.py               # Visualisation
│
├── configs/                           # YAML configuration files
│   ├── cql_default.yaml
│   ├── cql_alpha_sweep.yaml
│   ├── bc_baseline.yaml
│   └── experiment_grid.yaml
│
├── scripts/                           # Executable scripts
│   ├── 01_install_environment.py
│   ├── 02_collect_offline_data.py
│   ├── 03_train_cql.py
│   ├── 04_train_baselines.py
│   ├── 05_evaluate_policies.py
│   ├── 06_generate_figures.py
│   ├── reproduce_all.ps1             # Windows reproduction script
│   └── reproduce_all.sh              # macOS/Linux reproduction script
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_environment_exploration.ipynb
│   ├── 02_data_analysis.ipynb
│   ├── 03_cql_training.ipynb
│   ├── 04_results_visualization.ipynb
│   ├── 05_safety_analysis.ipynb
│   └── 06_policy_interpretation.ipynb
│
├── data/                              # Data directory
│   └── offline_datasets/
│
├── results/                           # Experimental results
│   ├── checkpoints/
│   ├── logs/
│   └── evaluation/
│
├── figures/                           # Generated figures
│
└── tests/                             # Unit tests
    ├── test_cql.py
    ├── test_environment.py
    ├── test_replay_buffer.py
    └── test_evaluation.py
```

## Quick Start

### Prerequisites

- Python 3.8+ (3.10 recommended)
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

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

Note: `01_install_environment.py` verifies all dependencies are correctly installed and tests the ICU-Sepsis environment.

### Reproduce All Experiments

To run the complete experimental pipeline (data collection, training, evaluation, and figure generation):

#### Windows (PowerShell)

```powershell
# Full reproduction
.\scripts\reproduce_all.ps1

# Quick test run
.\scripts\reproduce_all.ps1 -Quick
```

#### macOS / Linux (Bash)

```bash
# Full reproduction
bash scripts/reproduce_all.sh

# Quick test run
bash scripts/reproduce_all.sh --quick
```

## Experiments

### 1. Main Performance Comparison
Compare CQL against Behaviour Cloning, DQN, and random baselines on the ICU-Sepsis benchmark.

### 2. Alpha Sensitivity Analysis
Sweep over conservatism coefficient alpha in {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0} to analyse the trade-off between conservatism and optimality.

### 3. Data Efficiency Study
Evaluate performance with varying dataset sizes.

### 4. Safety Analysis
Analyse action distributions, extreme action frequencies, and comparison to clinician baselines.

## Algorithm Details

### Conservative Q-Learning (CQL)

CQL adds a conservative regularizer to standard Q-learning to prevent overestimation of Q-values for out-of-distribution actions:

$$\mathcal{L}_{CQL}(\theta) = \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q_\theta(s, a)) - \mathbb{E}_{a \sim \pi_\beta}[Q_\theta(s, a)] \right] + \frac{1}{2} \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (Q_\theta(s,a) - \mathcal{B}^\pi \hat{Q}_{\bar{\theta}}(s,a))^2 \right]$$

**Key Components:**
- **Conservative Penalty**: Pushes down Q-values for all actions while pushing up Q-values for actions in the dataset
- **TD Loss**: Standard temporal difference loss with Double DQN
- **Alpha**: Controls the degree of conservatism

### Action Space

The ICU-Sepsis environment has 25 discrete actions representing combinations of:
- **Vasopressors**: 5 levels (none, low, medium, high, very high)
- **IV Fluids**: 5 levels (none, low, medium, high, very high)

### State Space

716 discrete states representing patient physiological conditions including:
- Vital signs (heart rate, blood pressure, temperature)
- Lab values (lactate, creatinine, glucose)
- Demographics and comorbidities

## References

1. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. *NeurIPS*. — The original CQL algorithm implemented in this project.
2. Choudhary, S., et al. (2024). ICU-Sepsis: A Benchmark MDP Built from Real Medical Data. *RLC*. — The benchmark environment used for evaluation.
3. Komorowski, M., et al. (2018). The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. *Nature Medicine*. — Foundational work on RL for sepsis treatment that motivates this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This is a coursework project for educational purposes. The learned policies should not be used for actual medical decision-making without proper clinical validation and regulatory approval.
