#!/usr/bin/env python
"""
Script to install and verify the ICU-Sepsis environment.

This script:
1. Checks Python version
2. Installs required dependencies
3. Installs ICU-Sepsis environment
4. Verifies installation with a simple test

Usage:
    python scripts/01_install_environment.py
"""

import argparse
import os
import subprocess
import sys


def check_python_version():
    """Check that Python version is 3.8+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python 3.8+ required. Found: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")


def install_requirements():
    """Install requirements from requirements.txt"""
    print("\nInstalling requirements...")
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    requirements_path = os.path.join(project_root, "requirements.txt")
    
    if os.path.exists(requirements_path):
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", requirements_path
            ])
            print("✓ Requirements installed")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Some requirements failed to install: {e}")
    else:
        print(f"WARNING: requirements.txt not found at {requirements_path}")


def install_icu_sepsis():
    """Install the ICU-Sepsis environment"""
    print("\nInstalling ICU-Sepsis environment...")
    
    try:
        # Try to install from PyPI first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "icu-sepsis"
        ])
        print("✓ ICU-Sepsis installed from PyPI")
    except subprocess.CalledProcessError:
        print("PyPI installation failed, trying from GitHub...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/icu-sepsis/icu-sepsis.git"
            ])
            print("✓ ICU-Sepsis installed from GitHub")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: ICU-Sepsis installation failed: {e}")
            print("The project will use a mock environment for testing.")
            return False
    
    return True


def verify_installation(strict: bool) -> bool:
    """Verify the installation by running a simple test.

    Returns:
        True if critical checks pass, otherwise False.
    """
    print("\nVerifying installation...")
    ok = True
    
    # Test basic imports
    print("Testing imports...")
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy")
        ok = False
    
    try:
        import torch
        print(f"  ✓ torch (version {torch.__version__})")
        if torch.cuda.is_available():
            print(f"    CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("    CUDA not available, will use CPU")
    except ImportError:
        print("  ✗ torch")
        ok = False
    
    try:
        import gymnasium as gym
        print("  ✓ gymnasium")
    except ImportError:
        try:
            import gym
            print("  ✓ gym (legacy)")
        except ImportError:
            print("  ✗ gymnasium/gym")
            ok = False

    # Optional dependencies (do not fail setup)
    try:
        import yaml  # noqa: F401
        print("  ✓ pyyaml")
    except ImportError:
        print("  ⚠ pyyaml (optional)")

    try:
        import omegaconf  # noqa: F401
        print("  ✓ omegaconf")
    except ImportError:
        print("  ⚠ omegaconf (optional)")
    
    try:
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib")
    
    # Test ICU-Sepsis environment
    print("\nTesting ICU-Sepsis environment...")
    try:
        import icu_sepsis
        
        # Create environment directly (not via gym.make)
        base_env = icu_sepsis.ICUSepsisEnv()
        env = icu_sepsis.FlattenActionWrapper(base_env)
        state, info = env.reset(seed=42)
        
        print(f"  ✓ Environment created")
        print(f"    State space: {env.observation_space}")
        print(f"    Action space: {env.action_space}")
        print(f"    Initial state: {state}")
        
        # Take a few steps
        total_reward = 0
        for _ in range(5):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"    Test steps successful, total reward: {total_reward}")
        env.close()
        
    except Exception as e:
        print(f"  ✗ ICU-Sepsis environment not available: {e}")
        print("    Will use mock environment for development/testing")
        if strict:
            ok = False
    
    # Test project imports
    print("\nTesting project imports...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from src.algorithms import CQL, BehaviorCloning, DQN
        print("  ✓ Algorithms (CQL, BC, DQN)")
        
        from src.data import OfflineReplayBuffer
        print("  ✓ Data utilities")
        
        from src.environments import create_sepsis_env
        print("  ✓ Environment wrapper")
        
        from src.utils import evaluate_policy
        print("  ✓ Evaluation utilities")
        
    except Exception as e:
        print(f"  ✗ Project imports failed: {e}")
        ok = False

    return ok


def run_tests() -> bool:
    """Run the unit tests (pytest) if available."""
    print("\nRunning unit tests (pytest)...")
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)
        if result.returncode == 0:
            print("  ✓ Tests passed")
            return True
        print(f"  ✗ Tests failed (exit code {result.returncode})")
        return False
    except Exception as e:
        print(f"  ✗ Unable to run tests: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    directories = [
        "data/offline_datasets",
        "data/environment",
        "results/checkpoints",
        "results/logs",
        "results/figures",
    ]
    
    for directory in directories:
        dir_path = os.path.join(project_root, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ {directory}")


def main():
    """Main installation routine"""
    parser = argparse.ArgumentParser(description="Install and verify CQL-Sepsis environment")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Do not fail if ICU-Sepsis is unavailable (allows mock environment)",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run pytest after import checks",
    )
    args = parser.parse_args()

    strict = not args.non_strict

    print("=" * 60)
    print("CQL-Sepsis Environment Setup")
    print("=" * 60)
    
    check_python_version()
    install_requirements()
    install_icu_sepsis()
    create_directories()

    ok = verify_installation(strict=strict)
    if args.run_tests:
        ok = ok and run_tests()
    
    print("\n" + "=" * 60)
    if ok:
        print("Setup complete!")
    else:
        print("Setup completed with errors.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Collect offline data: python scripts/02_collect_offline_data.py")
    print("  2. Train CQL: python scripts/03_train_cql.py")
    print("  3. Evaluate: python scripts/05_evaluate_policies.py")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
