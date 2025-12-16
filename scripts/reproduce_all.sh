#!/bin/bash
# =============================================================================
# Reproduce All Experiments
# =============================================================================
# This script reproduces all experiments for the CQL Sepsis Treatment project.
# 
# Usage:
#   chmod +x scripts/reproduce_all.sh
#   ./scripts/reproduce_all.sh
#
# Options:
#   --quick      Run quick version with fewer iterations
#   --skip-data  Skip data collection (use existing data)
#   --skip-train Skip training (use existing checkpoints)
#
# Estimated time: ~4-6 hours for full reproduction
# =============================================================================

set -e  # Exit on error

# Parse arguments
QUICK=false
SKIP_DATA=false
SKIP_TRAIN=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
    esac
done

# Configuration
if [ "$QUICK" = true ]; then
    echo "Running in QUICK mode (reduced iterations)"
    N_ITERATIONS=10000
    EVAL_EPISODES=50
    DATA_EPISODES=5000
else
    N_ITERATIONS=100000
    EVAL_EPISODES=100
    DATA_EPISODES=50000
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  CQL Sepsis Treatment - Full Reproduction   ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Check Python environment
echo -e "${YELLOW}Checking Python environment...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}Found: $PYTHON_VERSION${NC}"

# =============================================================================
# Step 1: Environment Setup
# =============================================================================
echo ""
echo -e "${BLUE}Step 1: Environment Setup${NC}"
echo "----------------------------------------"

python scripts/01_install_environment.py

# =============================================================================
# Step 2: Data Collection
# =============================================================================
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo -e "${BLUE}Step 2: Collecting Offline Data (using EXPERT policy)${NC}"
    echo "----------------------------------------"
    
    # Use the real clinician expert policy for true offline RL
    python scripts/02_collect_offline_data.py \
        --policy_type expert \
        --n_episodes $DATA_EPISODES \
        --save_path data/offline_datasets/behavior_policy.pkl
else
    echo ""
    echo -e "${YELLOW}Step 2: Skipping data collection (--skip-data)${NC}"
fi

# =============================================================================
# Step 3: Train CQL with Different Alpha Values
# =============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo -e "${BLUE}Step 3: Training CQL Agents${NC}"
    echo "----------------------------------------"
    
    # Alpha sweep
    ALPHAS=(0.0 0.1 0.5 1.0 2.0 5.0 10.0)
    SEEDS=(42 123 456)
    
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo -e "${YELLOW}Training CQL with alpha=$alpha, seed=$seed${NC}"
            
            python scripts/03_train_cql.py \
                --config configs/cql_default.yaml \
                --dataset data/offline_datasets/behavior_policy.pkl \
                --alpha $alpha \
                --seed $seed \
                --n_iterations $N_ITERATIONS \
                --output_dir "results/cql_alpha_${alpha}_seed_${seed}"
        done
    done
    
    # =============================================================================
    # Step 4: Train Baselines
    # =============================================================================
    echo ""
    echo -e "${BLUE}Step 4: Training Baseline Models${NC}"
    echo "----------------------------------------"
    
    for seed in "${SEEDS[@]}"; do
        # Behavior Cloning
        echo -e "${YELLOW}Training BC with seed=$seed${NC}"
        python scripts/04_train_baselines.py \
            --algorithm bc \
            --dataset data/offline_datasets/behavior_policy.pkl \
            --seed $seed \
            --n_iterations $N_ITERATIONS \
            --output_dir "results/bc_seed_${seed}"
        
        # DQN (offline)
        echo -e "${YELLOW}Training DQN with seed=$seed${NC}"
        python scripts/04_train_baselines.py \
            --algorithm dqn \
            --dataset data/offline_datasets/behavior_policy.pkl \
            --seed $seed \
            --n_iterations $N_ITERATIONS \
            --output_dir "results/dqn_seed_${seed}"
    done
else
    echo ""
    echo -e "${YELLOW}Step 3-4: Skipping training (--skip-train)${NC}"
fi

# =============================================================================
# Step 5: Evaluation
# =============================================================================
echo ""
echo -e "${BLUE}Step 5: Evaluating All Policies${NC}"
echo "----------------------------------------"

python scripts/05_evaluate_policies.py \
    --checkpoints_dir results/ \
    --n_episodes $EVAL_EPISODES \
    --include_random \
    --safety_analysis

# =============================================================================
# Step 6: Generate Figures
# =============================================================================
echo ""
echo -e "${BLUE}Step 6: Generating Publication Figures${NC}"
echo "----------------------------------------"

python scripts/06_generate_figures.py \
    --results_dir results/ \
    --output_dir figures/ \
    --format pdf \
    --dpi 300

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  Reproduction Complete!                     ${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo "Results:"
echo "  - Checkpoints: results/"
echo "  - Evaluation:  results/evaluation/"
echo "  - Figures:     figures/"
echo ""
echo "Next steps:"
echo "  1. Review results in notebooks/"
echo "  2. Check figures/ for publication-ready plots"
echo "  3. Run 'python scripts/05_evaluate_policies.py' for detailed analysis"
echo ""

# Show summary statistics
if [ -f "results/evaluation/evaluation_results.json" ]; then
    echo "Quick Results Summary:"
    python -c "
import json
with open('results/evaluation/evaluation_results.json', 'r') as f:
    results = json.load(f)
for name, data in results.items():
    if 'mean_survival_rate' in data:
        print(f'  {name}: {data[\"mean_survival_rate\"]:.1%}')
"
fi
