"""
Unit tests for evaluation utilities.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.evaluation import (
    evaluate_policy,
    compute_confidence_intervals,
    compute_off_policy_estimates,
    analyze_safety_metrics,
)


class MockEnv:
    """Mock environment for testing."""
    
    def __init__(self, deterministic=True):
        self.deterministic = deterministic
        self.n_actions = 25
        self._step = 0
        
        # Mock spaces
        class MockSpace:
            n = 25
            def sample(self):
                return np.random.randint(0, 25)
        
        class MockObsSpace:
            n = 716
            shape = (1,)
        
        self.action_space = MockSpace()
        self.observation_space = MockObsSpace()
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._step = 0
        return np.array([0]), {}
    
    def step(self, action):
        self._step += 1
        
        # Terminal after random steps
        if self.deterministic:
            done = self._step >= 10
        else:
            done = np.random.random() < 0.1
        
        # Reward only at terminal
        if done:
            reward = 1.0 if np.random.random() > 0.5 else -1.0
        else:
            reward = 0.0
        
        next_state = np.array([self._step])
        return next_state, reward, done, False, {}


class MockPolicy:
    """Mock policy for testing."""
    
    def __init__(self, action=0):
        self.action = action
    
    def select_action(self, state, admissible_actions=None):
        return self.action
    
    def __call__(self, state, admissible_actions=None):
        return self.select_action(state, admissible_actions)


class TestEvaluatePolicy:
    """Tests for evaluate_policy function."""
    
    @pytest.fixture
    def env(self):
        return MockEnv()
    
    @pytest.fixture
    def policy(self):
        return MockPolicy()
    
    def test_basic_evaluation(self, env, policy):
        """Test basic policy evaluation."""
        results = evaluate_policy(
            env=env,
            policy=policy,
            n_episodes=10,
            seed=42,
        )
        
        assert 'survival_rate' in results
        assert 'mean_return' in results
        assert 'std_return' in results
        assert 'all_returns' in results
        assert 'all_lengths' in results
    
    def test_survival_rate_range(self, env, policy):
        """Test survival rate is in valid range."""
        results = evaluate_policy(
            env=env,
            policy=policy,
            n_episodes=50,
            seed=42,
        )
        
        assert 0 <= results['survival_rate'] <= 1
    
    def test_return_values(self, env, policy):
        """Test return values are valid."""
        results = evaluate_policy(
            env=env,
            policy=policy,
            n_episodes=20,
            seed=42,
        )
        
        # Returns should be in [-1, 1] for this environment
        returns = results['all_returns']
        assert all(-1 <= r <= 1 for r in returns)
    
    def test_episode_count(self, env, policy):
        """Test correct number of episodes."""
        n_episodes = 25
        results = evaluate_policy(
            env=env,
            policy=policy,
            n_episodes=n_episodes,
            seed=42,
        )
        
        assert len(results['all_returns']) == n_episodes
        assert len(results['all_lengths']) == n_episodes
    
    def test_callable_policy(self, env):
        """Test with callable policy."""
        def random_policy(state, admissible_actions=None):
            return np.random.randint(0, 25)
        
        results = evaluate_policy(
            env=env,
            policy=random_policy,
            n_episodes=10,
            seed=42,
        )
        
        assert 'survival_rate' in results
    
    def test_reproducibility(self, env, policy):
        """Test evaluation is reproducible with same seed."""
        results1 = evaluate_policy(env=env, policy=policy, n_episodes=10, seed=42)
        results2 = evaluate_policy(env=env, policy=policy, n_episodes=10, seed=42)
        
        assert results1['survival_rate'] == results2['survival_rate']
        assert results1['mean_return'] == results2['mean_return']


class TestConfidenceIntervals:
    """Tests for confidence interval computation."""
    
    def test_basic_ci(self):
        """Test basic confidence interval calculation."""
        data = np.random.normal(0.5, 0.1, 100)
        mean, ci_low, ci_high = compute_confidence_intervals(data)
        
        assert ci_low <= mean <= ci_high
        assert ci_low < ci_high
    
    def test_ci_with_known_distribution(self):
        """Test CI with known parameters."""
        np.random.seed(42)
        data = np.random.normal(0.8, 0.05, 1000)
        mean, ci_low, ci_high = compute_confidence_intervals(data)
        
        # Mean should be close to 0.8
        assert abs(mean - 0.8) < 0.01
        
        # CI should contain true mean
        assert ci_low < 0.8 < ci_high
    
    def test_ci_width(self):
        """Test CI width decreases with more samples."""
        np.random.seed(42)
        
        data_small = np.random.normal(0.5, 0.1, 10)
        data_large = np.random.normal(0.5, 0.1, 1000)
        
        _, low1, high1 = compute_confidence_intervals(data_small)
        _, low2, high2 = compute_confidence_intervals(data_large)
        
        width_small = high1 - low1
        width_large = high2 - low2
        
        # Larger sample should have narrower CI
        assert width_large < width_small
    
    def test_ci_with_single_value(self):
        """Test CI with single value."""
        data = np.array([0.5])
        mean, ci_low, ci_high = compute_confidence_intervals(data)
        
        assert mean == 0.5
        # CI should be degenerate or handle edge case


class TestOffPolicyEstimates:
    """Tests for off-policy evaluation estimates."""
    
    def test_importance_sampling(self):
        """Test importance sampling estimates."""
        # Create mock data
        trajectories = [
            {
                'states': [np.array([0]), np.array([1])],
                'actions': [0, 1],
                'rewards': [0, 1],
                'behavior_probs': [0.2, 0.3],
            }
        ]
        
        def target_policy(state):
            return 0.5  # Uniform
        
        estimate = compute_off_policy_estimates(
            trajectories=trajectories,
            target_policy=target_policy,
            gamma=0.99,
        )
        
        assert 'is_estimate' in estimate
        assert isinstance(estimate['is_estimate'], float)


class TestSafetyMetrics:
    """Tests for safety metric analysis."""
    
    def test_safety_analysis(self):
        """Test safety metric computation."""
        env = MockEnv()
        policy = MockPolicy()
        
        try:
            metrics = analyze_safety_metrics(
                policy=policy,
                env=env,
                n_episodes=10,
                seed=42,
            )
            
            # Should return safety-related metrics
            assert isinstance(metrics, dict)
        except NotImplementedError:
            # May not be implemented
            pass


class TestEdgeCases:
    """Tests for edge cases in evaluation."""
    
    def test_zero_episodes(self):
        """Test with zero episodes."""
        env = MockEnv()
        policy = MockPolicy()
        
        # Should handle gracefully or raise meaningful error
        try:
            results = evaluate_policy(env=env, policy=policy, n_episodes=0)
            assert results['survival_rate'] == 0 or len(results['all_returns']) == 0
        except ValueError:
            pass  # Expected
    
    def test_very_long_episode(self):
        """Test with potentially long episodes."""
        class LongEnv(MockEnv):
            def step(self, action):
                self._step += 1
                done = self._step >= 1000
                reward = 1.0 if done else 0.0
                return np.array([self._step]), reward, done, False, {}
        
        env = LongEnv()
        policy = MockPolicy()
        
        # Should complete without hanging
        results = evaluate_policy(
            env=env,
            policy=policy,
            n_episodes=2,
            seed=42,
        )
        
        assert len(results['all_returns']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
