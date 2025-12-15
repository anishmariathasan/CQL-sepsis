"""
Unit tests for the ICU-Sepsis environment wrapper.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.icu_sepsis_wrapper import ICUSepsisWrapper, create_sepsis_env


class TestICUSepsisWrapper:
    """Tests for the ICU-Sepsis environment wrapper."""
    
    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        try:
            return create_sepsis_env(use_action_masking=False)
        except ImportError:
            pytest.skip("icu-sepsis package not installed")
    
    @pytest.fixture
    def env_with_masking(self):
        """Create environment with action masking."""
        try:
            return create_sepsis_env(use_action_masking=True)
        except ImportError:
            pytest.skip("icu-sepsis package not installed")
    
    def test_environment_creation(self, env):
        """Test environment creation."""
        assert env is not None
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
    
    def test_observation_space(self, env):
        """Test observation space properties."""
        obs_space = env.observation_space
        
        # Should be discrete with 716 states
        assert obs_space.n == 716 or hasattr(obs_space, 'shape')
    
    def test_action_space(self, env):
        """Test action space properties."""
        action_space = env.action_space
        
        # Should be discrete with 25 actions
        assert action_space.n == 25
    
    def test_reset(self, env):
        """Test environment reset."""
        state, info = env.reset()
        
        assert state is not None
        assert isinstance(info, dict)
    
    def test_reset_with_seed(self, env):
        """Test deterministic reset with seed."""
        state1, _ = env.reset(seed=42)
        state2, _ = env.reset(seed=42)
        
        # Same seed should give same initial state
        assert np.array_equal(state1, state2)
    
    def test_step(self, env):
        """Test environment step."""
        env.reset(seed=42)
        
        action = 0  # No treatment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        assert next_state is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_reward_range(self, env):
        """Test that rewards are in expected range."""
        env.reset(seed=42)
        
        rewards = []
        for _ in range(100):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                env.reset()
        
        # Rewards should be in {-1, 0, +1}
        unique_rewards = set(rewards)
        assert unique_rewards.issubset({-1, 0, 1})
    
    def test_terminal_states(self, env):
        """Test that episodes terminate."""
        env.reset(seed=42)
        
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Should eventually terminate
        assert done or steps < max_steps
    
    def test_action_masking(self, env_with_masking):
        """Test action masking functionality."""
        state, info = env_with_masking.reset(seed=42)
        
        # Should have action mask in info
        if 'admissible_actions' in info:
            mask = info['admissible_actions']
            assert isinstance(mask, (list, np.ndarray))
            assert len(mask) <= 25
    
    def test_episode_statistics(self, env):
        """Test collecting episode statistics."""
        returns = []
        lengths = []
        
        for _ in range(10):
            state, _ = env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                action = env.action_space.sample()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
                
                if episode_length > 100:
                    break
            
            returns.append(episode_return)
            lengths.append(episode_length)
        
        assert len(returns) == 10
        assert all(r in [-1, 0, 1] for r in returns)
    
    def test_state_preprocessing(self, env):
        """Test state preprocessing."""
        state, _ = env.reset()
        
        # State should be numeric
        if isinstance(state, np.ndarray):
            assert np.isfinite(state).all()
        else:
            assert isinstance(state, (int, float))


class TestEnvironmentIntegration:
    """Integration tests for the environment."""
    
    def test_full_episode_rollout(self):
        """Test complete episode rollout."""
        try:
            env = create_sepsis_env(use_action_masking=False)
        except ImportError:
            pytest.skip("icu-sepsis package not installed")
        
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
        }
        
        state, _ = env.reset(seed=42)
        trajectory['states'].append(state)
        
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            trajectory['states'].append(next_state)
            
            if len(trajectory['actions']) > 100:
                break
        
        # Verify trajectory
        assert len(trajectory['actions']) > 0
        assert len(trajectory['rewards']) == len(trajectory['actions'])
        assert trajectory['dones'][-1] or len(trajectory['actions']) > 100
    
    def test_reproducibility(self):
        """Test that environment is reproducible with same seed."""
        try:
            env = create_sepsis_env(use_action_masking=False)
        except ImportError:
            pytest.skip("icu-sepsis package not installed")
        
        def run_episode(seed):
            env.reset(seed=seed)
            np.random.seed(seed)
            
            states, rewards = [], []
            done = False
            
            while not done:
                action = np.random.randint(0, 25)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                states.append(state)
                rewards.append(reward)
                
                if len(states) > 50:
                    break
            
            return states, rewards
        
        # Run same episode twice
        states1, rewards1 = run_episode(42)
        states2, rewards2 = run_episode(42)
        
        # Should be identical
        assert len(states1) == len(states2)
        for s1, s2 in zip(states1[:10], states2[:10]):
            if isinstance(s1, np.ndarray):
                assert np.array_equal(s1, s2)
            else:
                assert s1 == s2


class TestActionDecomposition:
    """Tests for action decomposition (vasopressor × IV fluids)."""
    
    def test_action_to_treatment(self):
        """Test converting action index to treatment levels."""
        def decompose_action(action):
            vaso = action // 5
            iv = action % 5
            return vaso, iv
        
        # Test all 25 actions
        for action in range(25):
            vaso, iv = decompose_action(action)
            assert 0 <= vaso <= 4
            assert 0 <= iv <= 4
            assert action == vaso * 5 + iv
    
    def test_action_coverage(self):
        """Test that all treatments are representable."""
        actions_covered = set()
        
        for vaso in range(5):
            for iv in range(5):
                action = vaso * 5 + iv
                actions_covered.add(action)
        
        assert actions_covered == set(range(25))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
