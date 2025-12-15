# =============================================================================
# Reproduce All Experiments (PowerShell Version)
# =============================================================================
# This script reproduces all experiments for the CQL Sepsis Treatment project.
# 
# Usage:
#   .\scripts\reproduce_all.ps1
#   .\scripts\reproduce_all.ps1 -Quick
#   .\scripts\reproduce_all.ps1 -SkipData
#   .\scripts\reproduce_all.ps1 -SkipTrain
#
# Estimated time: ~4-6 hours for full reproduction
# =============================================================================

param(
    [switch]$Quick,
    [switch]$SkipData,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"

# Configuration
if ($Quick) {
    Write-Host "Running in QUICK mode (reduced iterations)" -ForegroundColor Yellow
    $N_ITERATIONS = 10000
    $EVAL_EPISODES = 50
    $DATA_EPISODES = 5000
} else {
    $N_ITERATIONS = 100000
    $EVAL_EPISODES = 100
    $DATA_EPISODES = 50000
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  CQL Sepsis Treatment - Full Reproduction   " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python environment
Write-Host "Checking Python environment..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# =============================================================================
# Step 1: Environment Setup
# =============================================================================
Write-Host ""
Write-Host "Step 1: Environment Setup" -ForegroundColor Cyan
Write-Host "----------------------------------------"

python scripts/01_install_environment.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment setup failed!" -ForegroundColor Red
    exit 1
}

# =============================================================================
# Step 2: Data Collection
# =============================================================================
if (-not $SkipData) {
    Write-Host ""
    Write-Host "Step 2: Collecting Offline Data" -ForegroundColor Cyan
    Write-Host "----------------------------------------"
    
    python scripts/02_collect_offline_data.py --n_episodes $DATA_EPISODES --output_dir data/offline_datasets
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Data collection failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "Step 2: Skipping data collection (-SkipData)" -ForegroundColor Yellow
}

# =============================================================================
# Step 3: Train CQL with Different Alpha Values
# =============================================================================
if (-not $SkipTrain) {
    Write-Host ""
    Write-Host "Step 3: Training CQL Agents" -ForegroundColor Cyan
    Write-Host "----------------------------------------"
    
    $ALPHAS = @(0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    $SEEDS = @(42, 123, 456)
    
    foreach ($alpha in $ALPHAS) {
        foreach ($seed in $SEEDS) {
            Write-Host "Training CQL with alpha=$alpha, seed=$seed" -ForegroundColor Yellow
            
            python scripts/03_train_cql.py `
                --config configs/cql_default.yaml `
                --dataset data/offline_datasets/behavior_policy.pkl `
                --alpha $alpha `
                --seed $seed `
                --n_iterations $N_ITERATIONS `
                --output_dir "results/cql_alpha_${alpha}_seed_${seed}"
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "CQL training failed for alpha=$alpha, seed=$seed" -ForegroundColor Red
            }
        }
    }
    
    # =============================================================================
    # Step 4: Train Baselines
    # =============================================================================
    Write-Host ""
    Write-Host "Step 4: Training Baseline Models" -ForegroundColor Cyan
    Write-Host "----------------------------------------"
    
    foreach ($seed in $SEEDS) {
        # Behavior Cloning
        Write-Host "Training BC with seed=$seed" -ForegroundColor Yellow
        python scripts/04_train_baselines.py `
            --algorithm bc `
            --dataset data/offline_datasets/behavior_policy.pkl `
            --seed $seed `
            --n_iterations $N_ITERATIONS `
            --output_dir "results/bc_seed_${seed}"
        
        # DQN (offline)
        Write-Host "Training DQN with seed=$seed" -ForegroundColor Yellow
        python scripts/04_train_baselines.py `
            --algorithm dqn `
            --dataset data/offline_datasets/behavior_policy.pkl `
            --seed $seed `
            --n_iterations $N_ITERATIONS `
            --output_dir "results/dqn_seed_${seed}"
    }
} else {
    Write-Host ""
    Write-Host "Step 3-4: Skipping training (-SkipTrain)" -ForegroundColor Yellow
}

# =============================================================================
# Step 5: Evaluation
# =============================================================================
Write-Host ""
Write-Host "Step 5: Evaluating All Policies" -ForegroundColor Cyan
Write-Host "----------------------------------------"

python scripts/05_evaluate_policies.py `
    --checkpoints_dir results/ `
    --n_episodes $EVAL_EPISODES `
    --include_random `
    --safety_analysis

# =============================================================================
# Step 6: Generate Figures
# =============================================================================
Write-Host ""
Write-Host "Step 6: Generating Publication Figures" -ForegroundColor Cyan
Write-Host "----------------------------------------"

python scripts/06_generate_figures.py `
    --results_dir results/ `
    --output_dir figures/ `
    --format pdf `
    --dpi 300

# =============================================================================
# Summary
# =============================================================================
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  Reproduction Complete!                     " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results:"
Write-Host "  - Checkpoints: results/"
Write-Host "  - Evaluation:  results/evaluation/"
Write-Host "  - Figures:     figures/"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Review results in notebooks/"
Write-Host "  2. Check figures/ for publication-ready plots"
Write-Host "  3. Run 'python scripts/05_evaluate_policies.py' for detailed analysis"
Write-Host ""

# Show summary statistics
$evalFile = "results/evaluation/evaluation_results.json"
if (Test-Path $evalFile) {
    Write-Host "Quick Results Summary:"
    python -c @"
import json
with open('$evalFile', 'r') as f:
    results = json.load(f)
for name, data in results.items():
    if 'mean_survival_rate' in data:
        print(f'  {name}: {data[\"mean_survival_rate\"]:.1%}')
"@
}
