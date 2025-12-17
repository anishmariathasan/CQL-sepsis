"""
Wrapper for the ICU-Sepsis Gymnasium environment.

This module provides a wrapper around the ICU-Sepsis benchmark environment
with additional utilities for analysis and visualisation.

The ICU-Sepsis environment simulates sepsis treatment in the ICU:
- State space: 716 discrete states (patient physiological conditions)
- Action space: 25 discrete actions (5x5 grid of vasopressors x IV fluids)
- Reward: +1 for patient survival, 0 otherwise
- Terminal: Episode ends at patient discharge or death

References:
    - Choudhary et al. "ICU-Sepsis: A Benchmark MDP", RLC 2024
    - https://github.com/icu-sepsis/icu-sepsis
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import Wrapper
except ImportError:
    import gym
    from gym import Wrapper

logger = logging.getLogger(__name__)

# Action descriptions (5x5 grid: vasopressor x IV fluid levels)
VASOPRESSOR_LEVELS = ["None", "Low", "Medium", "High", "Very High"]
IV_FLUID_LEVELS = ["None", "Low", "Medium", "High", "Very High"]

# Number of states and actions
N_STATES = 716
N_ACTIONS = 25


def get_action_description(action: int) -> Dict[str, str]:
    """
    Get human-readable description of an action.
    
    The 25 actions represent a 5x5 grid of:
        - Vasopressor levels (0-4): None, Low, Medium, High, Very High
        - IV Fluid levels (0-4): None, Low, Medium, High, Very High
    
    Action index = vasopressor_level * 5 + iv_fluid_level
    
    Args:
        action: Action index (0-24)
    
    Returns:
        Dictionary with 'vasopressor', 'iv_fluid', and 'description' keys
    """
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"Invalid action: {action}. Must be in [0, {N_ACTIONS})")
    
    vasopressor_level = action // 5
    iv_fluid_level = action % 5
    
    return {
        "action_id": action,
        "vasopressor": VASOPRESSOR_LEVELS[vasopressor_level],
        "vasopressor_level": vasopressor_level,
        "iv_fluid": IV_FLUID_LEVELS[iv_fluid_level],
        "iv_fluid_level": iv_fluid_level,
        "description": f"Vasopressor: {VASOPRESSOR_LEVELS[vasopressor_level]}, "
                       f"IV Fluid: {IV_FLUID_LEVELS[iv_fluid_level]}",
    }


def get_state_description(state: int) -> Dict[str, Any]:
    """
    Get description of a state.
    
    Note: The exact state representation in ICU-Sepsis is a discretization
    of continuous patient features. This function provides basic info.
    
    Args:
        state: State index (0-715)
    
    Returns:
        Dictionary with state information
    """
    if not 0 <= state < N_STATES:
        raise ValueError(f"Invalid state: {state}. Must be in [0, {N_STATES})")
    
    return {
        "state_id": state,
        "n_states": N_STATES,
        "description": f"State {state} of {N_STATES}",
    }


def action_to_doses(action: int) -> Tuple[int, int]:
    """
    Convert action index to (vasopressor_level, iv_fluid_level).
    
    Args:
        action: Action index (0-24)
    
    Returns:
        Tuple of (vasopressor_level, iv_fluid_level), each in [0, 4]
    """
    vasopressor_level = action // 5
    iv_fluid_level = action % 5
    return vasopressor_level, iv_fluid_level


def doses_to_action(vasopressor_level: int, iv_fluid_level: int) -> int:
    """
    Convert dose levels to action index.
    
    Args:
        vasopressor_level: Vasopressor level (0-4)
        iv_fluid_level: IV fluid level (0-4)
    
    Returns:
        Action index (0-24)
    """
    if not 0 <= vasopressor_level < 5:
        raise ValueError(f"Invalid vasopressor level: {vasopressor_level}")
    if not 0 <= iv_fluid_level < 5:
        raise ValueError(f"Invalid IV fluid level: {iv_fluid_level}")
    
    return vasopressor_level * 5 + iv_fluid_level


class ICUSepsisWrapper(Wrapper):
    """
    Wrapper for the ICU-Sepsis environment with additional utilities.
    
    Features:
        - Consistent interface for state/action spaces
        - Action masking support
        - Logging and statistics tracking
        - State/action description utilities
    
    Args:
        env: Base ICU-Sepsis environment
        use_action_masking: Whether to enforce admissible actions
        log_episodes: Whether to log episode statistics
    
    Example:
        >>> env = ICUSepsisWrapper(gym.make('Sepsis-v0'))
        >>> state, info = env.reset()
        >>> action = env.action_space.sample()
        >>> next_state, reward, done, truncated, info = env.step(action)
    """
    
    def __init__(
        self,
        env: gym.Env,
        use_action_masking: bool = True,
        log_episodes: bool = False,
    ):
        super().__init__(env)
        
        self.use_action_masking = use_action_masking
        self.log_episodes = log_episodes
        
        # Statistics tracking
        self.n_episodes = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        
        # Current episode tracking
        self._current_episode_return = 0.0
        self._current_episode_length = 0
        self._current_episode_actions: List[int] = []
        
        logger.info(f"Initialized ICUSepsisWrapper")
        logger.info(f"  State space: {N_STATES} discrete states")
        logger.info(f"  Action space: {N_ACTIONS} discrete actions")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment."""
        # Log previous episode statistics
        if self._current_episode_length > 0 and self.log_episodes:
            self.episode_returns.append(self._current_episode_return)
            self.episode_lengths.append(self._current_episode_length)
            self.n_episodes += 1
            
            if self.n_episodes % 100 == 0:
                avg_return = np.mean(self.episode_returns[-100:])
                logger.info(
                    f"Episode {self.n_episodes}: "
                    f"Return = {self._current_episode_return:.1f}, "
                    f"Length = {self._current_episode_length}, "
                    f"Avg Return (last 100) = {avg_return:.3f}"
                )
        
        # Reset tracking
        self._current_episode_return = 0.0
        self._current_episode_length = 0
        self._current_episode_actions = []
        
        # Reset base environment
        state, info = self.env.reset(seed=seed, options=options)
        
        # Add custom info
        info["state_description"] = get_state_description(state)
        
        return state, info
    
    def step(
        self,
        action: int,
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Validate action
        if self.use_action_masking:
            admissible = self.get_admissible_actions()
            if action not in admissible:
                # Map to nearest admissible action
                logger.warning(
                    f"Action {action} not admissible. "
                    f"Admissible: {admissible}"
                )
                action = admissible[0] if admissible else 0
        
        # Take step
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Update tracking
        self._current_episode_return += reward
        self._current_episode_length += 1
        self._current_episode_actions.append(action)
        
        # Add custom info
        info["action_description"] = get_action_description(action)
        info["state_description"] = get_state_description(next_state)
        info["episode_return"] = self._current_episode_return
        info["episode_length"] = self._current_episode_length
        
        return next_state, reward, terminated, truncated, info
    
    def get_admissible_actions(self) -> List[int]:
        """
        Get list of admissible actions for current state.
        
        Returns:
            List of admissible action indices
        """
        try:
            return list(self.env.get_admissible_actions())
        except AttributeError:
            # If not available, all actions are admissible
            return list(range(N_ACTIONS))
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask as boolean array.
        
        Returns:
            Boolean array of shape (N_ACTIONS,) where True = admissible
        """
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[self.get_admissible_actions()] = True
        return mask
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collected episode statistics."""
        if len(self.episode_returns) == 0:
            return {"n_episodes": 0}
        
        returns = np.array(self.episode_returns)
        lengths = np.array(self.episode_lengths)
        
        return {
            "n_episodes": self.n_episodes,
            "avg_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "survival_rate": float(np.mean(returns > 0)),
            "avg_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
        }
    
    def describe_action(self, action: int) -> str:
        """Get human-readable action description."""
        desc = get_action_description(action)
        return desc["description"]
    
    @property
    def n_states(self) -> int:
        """Number of discrete states."""
        return N_STATES
    
    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return N_ACTIONS


def create_sepsis_env(
    use_action_masking: bool = True,
    log_episodes: bool = False,
    **kwargs,
) -> ICUSepsisWrapper:
    """
    Create an ICU-Sepsis environment with wrapper.
    
    Args:
        use_action_masking: Whether to enforce admissible actions
        log_episodes: Whether to log episode statistics
        **kwargs: Additional arguments for gym.make
    
    Returns:
        Wrapped ICU-Sepsis environment
    
    Example:
        >>> env = create_sepsis_env()
        >>> state, info = env.reset(seed=42)
        >>> print(f"State: {state}, Admissible actions: {env.get_admissible_actions()}")
    """
    try:
        # Create the real ICU-Sepsis environment directly
        import icu_sepsis
        base_env = icu_sepsis.ICUSepsisEnv()
        # Flatten action space from MultiDiscrete([5,5]) to Discrete(25)
        base_env = icu_sepsis.FlattenActionWrapper(base_env)
        logger.info("Created ICU-Sepsis environment with FlattenActionWrapper")
    except ImportError as e:
        logger.warning(f"Could not import icu_sepsis: {e}")
        logger.warning("Using mock environment for testing")
        base_env = MockSepsisEnv()
    except Exception as e:
        logger.warning(f"Could not create ICU-Sepsis environment: {e}")
        logger.warning("Using mock environment for testing")
        base_env = MockSepsisEnv()
    
    return ICUSepsisWrapper(
        env=base_env,
        use_action_masking=use_action_masking,
        log_episodes=log_episodes,
    )


class MockSepsisEnv(gym.Env):
    """
    Mock ICU-Sepsis environment for testing when real env is unavailable.
    
    Simulates the basic structure of the ICU-Sepsis environment with
    meaningful dynamics where actions affect outcomes.
    
    State representation (simplified):
        - States 0-179: Low severity (better prognosis)
        - States 180-537: Medium severity 
        - States 538-715: High severity (worse prognosis)
    
    Action effects:
        - Moderate vasopressors (levels 1-2) and IV fluids (levels 1-3) are generally beneficial
        - Extreme actions (none or very high) can be harmful
        - Optimal treatment depends on patient severity
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Discrete(N_STATES)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        
        self.state = 0
        self.step_count = 0
        self.max_steps = 20
        self._np_random = None
        self._severity = 0  # 0=low, 1=medium, 2=high
        self._patient_health = 1.0  # Track patient health trajectory
    
    def _get_severity(self, state):
        """Get severity level from state."""
        if state < 180:
            return 0  # Low severity
        elif state < 538:
            return 1  # Medium severity
        else:
            return 2  # High severity
    
    def _action_effect(self, action, severity):
        """
        Compute the effect of an action given patient severity.
        
        Returns a value in [-0.3, 0.3] indicating health change.
        Positive = health improves, Negative = health worsens.
        """
        vaso_level = action // 5  # 0-4
        fluid_level = action % 5  # 0-4
        
        effect = 0.0
        
        # Optimal treatment varies by severity
        if severity == 0:  # Low severity - minimal intervention is best
            # Ideal: low vasopressors (0-1), moderate fluids (1-2)
            if vaso_level <= 1 and 1 <= fluid_level <= 2:
                effect = 0.15
            elif vaso_level >= 3:  # Too much vasopressor is bad
                effect = -0.2
            elif fluid_level >= 4:  # Too much fluid is bad
                effect = -0.15
            else:
                effect = 0.05
                
        elif severity == 1:  # Medium severity - balanced treatment
            # Ideal: moderate vasopressors (1-2), moderate fluids (1-3)
            if 1 <= vaso_level <= 2 and 1 <= fluid_level <= 3:
                effect = 0.2
            elif vaso_level == 0 and fluid_level == 0:  # No treatment is bad
                effect = -0.25
            elif vaso_level >= 4 or fluid_level >= 4:  # Too aggressive
                effect = -0.1
            else:
                effect = 0.0
                
        else:  # High severity - aggressive treatment needed
            # Ideal: higher vasopressors (2-3), higher fluids (2-4)
            if 2 <= vaso_level <= 3 and 2 <= fluid_level <= 4:
                effect = 0.25
            elif vaso_level <= 1 and fluid_level <= 1:  # Too conservative
                effect = -0.3
            elif vaso_level == 4 and fluid_level == 4:  # Max everything - risky
                effect = -0.1 + 0.15 * np.random.random()  # High variance
            else:
                effect = 0.05
        
        # Add some stochasticity
        effect += np.random.normal(0, 0.05)
        
        return np.clip(effect, -0.3, 0.3)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.state = np.random.randint(0, N_STATES)
        self.step_count = 0
        self._severity = self._get_severity(self.state)
        self._patient_health = 0.5 + 0.3 * np.random.random()  # Start between 0.5-0.8
        
        # Lower severity patients start healthier
        if self._severity == 0:
            self._patient_health += 0.15
        elif self._severity == 2:
            self._patient_health -= 0.15
        
        return self.state, {}
    
    def step(self, action):
        self.step_count += 1
        
        # Compute action effect on health
        health_change = self._action_effect(action, self._severity)
        self._patient_health = np.clip(self._patient_health + health_change, 0.0, 1.0)
        
        # Transition to next state (correlated with current state and health)
        if self._patient_health > 0.6:
            # Improving - tend toward lower severity states
            next_state = max(0, self.state - np.random.randint(0, 50))
        elif self._patient_health < 0.3:
            # Worsening - tend toward higher severity states  
            next_state = min(N_STATES - 1, self.state + np.random.randint(0, 100))
        else:
            # Stable - stay in similar range
            next_state = np.clip(
                self.state + np.random.randint(-30, 30), 
                0, N_STATES - 1
            )
        
        # Update severity based on new state
        self._severity = self._get_severity(next_state)
        
        # Episode termination
        # - Ends if health drops too low (death) or gets high enough (recovery)
        # - Or at max steps
        terminated = False
        reward = 0.0
        
        if self._patient_health <= 0.1:
            # Patient dies
            terminated = True
            reward = 0.0
        elif self._patient_health >= 0.9:
            # Patient recovers
            terminated = True
            reward = 1.0
        elif self.step_count >= self.max_steps:
            # Episode timeout - outcome based on final health
            terminated = True
            # Probabilistic survival based on health
            survival_prob = self._patient_health
            reward = 1.0 if np.random.random() < survival_prob else 0.0
        
        self.state = next_state
        
        return next_state, reward, terminated, False, {}
    
    def get_admissible_actions(self):
        """
        Return admissible actions based on patient state.
        
        In high severity, more aggressive options are available.
        In low severity, extreme treatments may not be indicated.
        """
        if self._severity == 0:  # Low severity - restrict aggressive treatments
            # Exclude very high vasopressors (level 4)
            admissible = [a for a in range(N_ACTIONS) if a // 5 < 4]
        elif self._severity == 2:  # High severity - all treatments available
            admissible = list(range(N_ACTIONS))
        else:  # Medium - most treatments available
            admissible = list(range(N_ACTIONS))
        
        return admissible
    
    def render(self):
        pass
    
    def close(self):
        pass


if __name__ == "__main__":
    # Test the wrapper
    logging.basicConfig(level=logging.INFO)
    
    # Create environment
    env = create_sepsis_env(log_episodes=True)
    
    # Run a few episodes
    for ep in range(5):
        state, info = env.reset(seed=ep)
        done = False
        
        while not done:
            # Get admissible actions
            admissible = env.get_admissible_actions()
            action = np.random.choice(admissible)
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        print(f"Episode {ep + 1}: Return = {info['episode_return']}, "
              f"Length = {info['episode_length']}")
    
    # Print statistics
    print(f"\nStatistics: {env.get_statistics()}")
    
    # Test action descriptions
    print("\nAction descriptions:")
    for action in [0, 6, 12, 18, 24]:
        print(f"  Action {action}: {env.describe_action(action)}")
    
    print("\nEnvironment wrapper tests passed!")
