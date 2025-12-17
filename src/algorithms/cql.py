"""
Conservative Q-Learning (CQL) for Offline Reinforcement Learning.

This module implements Conservative Q-Learning as described in:
Kumar et al. "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS 2020

Key Features:
- Conservative Q-function penalty to avoid overestimation
- Supports discrete action spaces (ICU-Sepsis: 25 actions)
- Double DQN architecture for stability
- Soft target network updates
- Support for variable alpha (conservatism coefficient)

References:
    - https://arxiv.org/abs/2006.04779
    - https://github.com/aviralkumar2907/CQL
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """
    Q-Network for discrete action spaces.
    
    Architecture:
        Input -> [Linear -> ReLU] x num_layers -> Linear -> Output
    
    Args:
        state_dim: Dimension of state space (or number of discrete states)
        action_dim: Number of discrete actions
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        use_one_hot: Whether to use one-hot encoding for discrete states
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_one_hot: bool = False,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_one_hot = use_one_hot
        
        # Input dimension depends on state encoding
        input_dim = state_dim if use_one_hot else hidden_dim
        
        # Embedding layer for discrete states (if not using one-hot)
        if not use_one_hot:
            self.state_embedding = nn.Embedding(state_dim, hidden_dim)
        
        # Build MLP layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialise weights
        self._initialise_weights()
    
    def _initialise_weights(self):
        """Initialise network weights using Xavier initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions given states.
        
        Args:
            states: State tensor of shape (batch_size,) for discrete states
                   or (batch_size, state_dim) for one-hot encoded states
        
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        if self.use_one_hot:
            # States are already one-hot encoded
            x = states.float()
        else:
            # Embed discrete state indices
            states = states.long()
            x = self.state_embedding(states)
        
        return self.network(x)
    
    def get_action(
        self,
        states: torch.Tensor,
        admissible_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get greedy action for given states.
        
        Args:
            states: State tensor
            admissible_actions: Optional mask of admissible actions (1 = admissible, 0 = not)
        
        Returns:
            Action indices tensor of shape (batch_size,)
        """
        q_values = self.forward(states)
        
        if admissible_actions is not None:
            # Mask out inadmissible actions with large negative value
            q_values = q_values.masked_fill(~admissible_actions.bool(), float('-inf'))
        
        return q_values.argmax(dim=1)


class CQL:
    """
    Conservative Q-Learning for Offline Reinforcement Learning.
    
    CQL adds a conservative regulariser to standard Q-learning to prevent
    overestimation of Q-values for out-of-distribution (OOD) actions. The
    key insight is to learn a Q-function that lower-bounds the true Q-function
    for OOD actions while being accurate for in-distribution actions.
    
    Loss function:
        L_CQL = L_TD + alpha * (E_s[logsumexp Q(s,a)] - E_{s,a~D}[Q(s,a)])
    
    where:
        - L_TD: Standard temporal difference loss
        - alpha: Conservatism coefficient
        - First term: Pushes down Q-values for all actions
        - Second term: Pushes up Q-values for dataset actions
    
    Args:
        state_dim: Number of discrete states
        action_dim: Number of discrete actions
        hidden_dim: Q-network hidden layer dimension
        num_layers: Number of hidden layers in Q-network
        lr: Learning rate for optimizer
        alpha: CQL conservatism coefficient (higher = more conservative)
        gamma: Discount factor
        tau: Soft target update rate
        use_double_dqn: Whether to use Double DQN for target computation
        use_one_hot: Whether to use one-hot state encoding
        grad_clip: Gradient clipping norm (None to disable)
        device: Device to run on ('cuda' or 'cpu')
    
    Example:
        >>> cql = CQL(state_dim=716, action_dim=25, alpha=1.0)
        >>> metrics = cql.update(batch)
        >>> action = cql.select_action(state)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        lr: float = 3e-4,
        alpha: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        use_double_dqn: bool = True,
        use_one_hot: bool = False,
        grad_clip: Optional[float] = 1.0,
        device: str = "auto",
    ):
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Store hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.use_double_dqn = use_double_dqn
        self.use_one_hot = use_one_hot
        self.grad_clip = grad_clip
        
        # Initialise Q-networks
        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_one_hot=use_one_hot,
        ).to(self.device)
        
        # Target network (copy of Q-network)
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()
        
        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Optimizer (PyTorch API)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Training statistics
        self.training_step = 0
        
        logger.info(f"Initialised CQL agent on {self.device}")
        logger.info(f"  State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"  Alpha: {alpha}, Gamma: {gamma}, Tau: {tau}")
        logger.info(f"  Network: {hidden_dim} hidden units, {num_layers} layers")
    
    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CQL loss = TD loss + alpha * CQL penalty.
        
        The CQL penalty pushes down Q-values for all actions while pushing up
        Q-values for actions in the dataset, resulting in a conservative
        Q-function estimate.
        
        Args:
            states: Batch of states, shape (batch_size,) or (batch_size, state_dim)
            actions: Batch of actions, shape (batch_size,)
            rewards: Batch of rewards, shape (batch_size,)
            next_states: Batch of next states
            dones: Batch of done flags, shape (batch_size,)
        
        Returns:
            loss: Total loss tensor for backward pass
            metrics: Dictionary of logging metrics including:
                - td_loss: Temporal difference loss
                - cql_penalty: CQL regularization term
                - total_loss: Combined loss
                - q_values_mean: Mean Q-value for dataset actions
                - q_values_logsumexp: Mean logsumexp of Q-values
        """
        batch_size = states.shape[0]
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device).float()
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).float()
        
        # Reshape if necessary
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # ========== Compute TD Loss (Standard DQN) ==========
        
        # Get current Q-values for actions taken
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions)
        
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: Use online network to select actions
                next_q_values = self.q_network(next_states)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                
                # Use target network to evaluate selected actions
                next_target_q_values = self.target_network(next_states)
                next_q = next_target_q_values.gather(1, next_actions)
            else:
                # Standard DQN: Use target network for both
                next_target_q_values = self.target_network(next_states)
                next_q = next_target_q_values.max(dim=1, keepdim=True)[0]
            
            # Compute TD targets
            td_targets = rewards + (1 - dones) * self.gamma * next_q
        
        # TD loss (MSE)
        td_loss = F.mse_loss(current_q, td_targets)
        
        # ========== Compute CQL Penalty ==========
        
        # Push down Q-values for all actions: logsumexp(Q(s, a))
        # This is a soft approximation to max_a Q(s, a)
        logsumexp_q = torch.logsumexp(current_q_values, dim=1, keepdim=True)
        
        # Push up Q-values for dataset actions: Q(s, a_data)
        data_q = current_q
        
        # CQL penalty: E_s[logsumexp Q(s,a)] - E_{s,a~D}[Q(s,a)]
        cql_penalty = (logsumexp_q - data_q).mean()
        
        # ========== Total Loss ==========
        
        total_loss = td_loss + self.alpha * cql_penalty
        
        # ========== Logging Metrics ==========
        
        metrics = {
            "td_loss": td_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "total_loss": total_loss.item(),
            "q_values_mean": current_q.mean().item(),
            "q_values_std": current_q.std().item(),
            "q_values_max": current_q.max().item(),
            "q_values_min": current_q.min().item(),
            "logsumexp_mean": logsumexp_q.mean().item(),
            "td_targets_mean": td_targets.mean().item(),
        }
        
        return total_loss, metrics
    
    def update(
        self,
        batch: Tuple[
            Union[np.ndarray, torch.Tensor],
            Union[np.ndarray, torch.Tensor],
            Union[np.ndarray, torch.Tensor],
            Union[np.ndarray, torch.Tensor],
            Union[np.ndarray, torch.Tensor],
        ],
    ) -> Dict[str, float]:
        """
        Perform a single gradient update step.
        
        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
        
        Returns:
            metrics: Dictionary of training metrics
        """
        # Unpack batch
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors if necessary
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards)
        if isinstance(next_states, np.ndarray):
            next_states = torch.from_numpy(next_states)
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones)
        
        # Set network to training mode
        self.q_network.train()
        
        # Compute loss
        loss, metrics = self.compute_cql_loss(
            states, actions, rewards, next_states, dones
        )
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.grad_clip
            )
            metrics["grad_norm"] = grad_norm.item()
        
        self.optimizer.step()
        
        # Soft target update
        self._soft_update_target()
        
        # Update training step
        self.training_step += 1
        metrics["training_step"] = self.training_step
        
        return metrics
    
    def _soft_update_target(self):
        """Soft update target network: θ_target = τ*θ + (1-τ)*θ_target"""
        for param, target_param in zip(
            self.q_network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def select_action(
        self,
        state: Union[int, np.ndarray, torch.Tensor],
        eval_mode: bool = True,
        admissible_actions: Optional[Union[List[int], np.ndarray]] = None,
    ) -> int:
        """
        Select action using greedy policy.
        
        Args:
            state: Current state (integer index or array)
            eval_mode: Whether in evaluation mode (always greedy)
            admissible_actions: Optional list of admissible action indices
        
        Returns:
            Selected action index
        """
        # Convert state to tensor
        if isinstance(state, int):
            state = torch.tensor([state])
        elif isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        
        if state.dim() == 0:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        # Set network to eval mode
        self.q_network.eval()
        
        with torch.no_grad():
            q_values = self.q_network(state)
            
            if admissible_actions is not None:
                # Create mask for admissible actions
                mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
                mask[admissible_actions] = True
                q_values = q_values.masked_fill(~mask.unsqueeze(0), float('-inf'))
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def get_q_values(
        self,
        states: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Get Q-values for given states.
        
        Args:
            states: States to evaluate, shape (batch_size,) or (batch_size, state_dim)
        
        Returns:
            Q-values array of shape (batch_size, action_dim)
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        
        states = states.to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(states)
        
        return q_values.cpu().numpy()
    
    def save(self, filepath: str):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "hyperparameters": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "lr": self.lr,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "tau": self.tau,
                "use_double_dqn": self.use_double_dqn,
                "use_one_hot": self.use_one_hot,
            },
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
        
        logger.info(f"Loaded checkpoint from {filepath}")
        logger.info(f"  Training step: {self.training_step}")
    
    @classmethod
    def from_checkpoint(cls, filepath: str, device: str = "auto") -> "CQL":
        """
        Create CQL agent from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
        
        Returns:
            CQL agent with loaded weights
        """
        checkpoint = torch.load(filepath, map_location="cpu")
        hyperparameters = checkpoint["hyperparameters"]
        
        agent = cls(
            state_dim=hyperparameters["state_dim"],
            action_dim=hyperparameters["action_dim"],
            hidden_dim=hyperparameters["hidden_dim"],
            num_layers=hyperparameters["num_layers"],
            lr=hyperparameters["lr"],
            alpha=hyperparameters["alpha"],
            gamma=hyperparameters["gamma"],
            tau=hyperparameters["tau"],
            use_double_dqn=hyperparameters["use_double_dqn"],
            use_one_hot=hyperparameters["use_one_hot"],
            device=device,
        )
        agent.load(filepath)
        
        return agent
    
    def set_alpha(self, alpha: float):
        """
        Update the conservatism coefficient.
        
        Args:
            alpha: New alpha value
        """
        self.alpha = alpha
        logger.info(f"Updated alpha to {alpha}")


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    # Create CQL agent
    agent = CQL(
        state_dim=716,
        action_dim=25,
        hidden_dim=256,
        alpha=1.0,
    )
    
    # Test with dummy batch
    batch_size = 32
    states = torch.randint(0, 716, (batch_size,))
    actions = torch.randint(0, 25, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randint(0, 716, (batch_size,))
    dones = torch.zeros(batch_size)
    
    batch = (states, actions, rewards, next_states, dones)
    
    # Test update
    metrics = agent.update(batch)
    print("Update metrics:", metrics)
    
    # Test action selection
    state = 100
    action = agent.select_action(state)
    print(f"Selected action for state {state}: {action}")
    
    # Test save/load
    agent.save("test_checkpoint.pt")
    agent2 = CQL.from_checkpoint("test_checkpoint.pt")
    print("Checkpoint test passed!")
    
    # Clean up
    import os
    os.remove("test_checkpoint.pt")
