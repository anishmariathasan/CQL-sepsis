"""
Deep Q-Network (DQN) baseline for comparison.

This module implements a standard DQN for offline RL (without CQL penalty).
It serves as a baseline to demonstrate the value of conservative estimation.

Note: Using standard DQN on offline data typically leads to overestimation
of Q-values for out-of-distribution actions, resulting in poor performance.
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
    """Q-Network for discrete action spaces."""
    
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
        self.use_one_hot = use_one_hot
        
        input_dim = state_dim if use_one_hot else hidden_dim
        
        if not use_one_hot:
            self.state_embedding = nn.Embedding(state_dim, hidden_dim)
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self.use_one_hot:
            x = states.float()
        else:
            states = states.long()
            x = self.state_embedding(states)
        return self.network(x)


class DQN:
    """
    Deep Q-Network for offline reinforcement learning (baseline).
    
    This is standard DQN without the CQL penalty. When used on offline data,
    it typically suffers from overestimation of Q-values for OOD actions.
    
    Args:
        state_dim: Number of discrete states
        action_dim: Number of discrete actions
        hidden_dim: Network hidden dimension
        num_layers: Number of hidden layers
        lr: Learning rate
        gamma: Discount factor
        tau: Soft target update rate
        use_double_dqn: Whether to use Double DQN
        use_one_hot: Whether to use one-hot state encoding
        grad_clip: Gradient clipping norm
        device: Device to run on
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        use_double_dqn: bool = True,
        use_one_hot: bool = False,
        grad_clip: Optional[float] = 1.0,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.use_double_dqn = use_double_dqn
        self.use_one_hot = use_one_hot
        self.grad_clip = grad_clip
        
        # Initialize networks
        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_one_hot=use_one_hot,
        ).to(self.device)
        
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()
        
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.training_step = 0
        
        logger.info(f"Initialized DQN agent on {self.device}")
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute standard TD loss."""
        states = states.to(self.device)
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device).float()
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).float()
        
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Current Q-values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions)
        
        with torch.no_grad():
            if self.use_double_dqn:
                next_q_values = self.q_network(next_states)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                next_target_q_values = self.target_network(next_states)
                next_q = next_target_q_values.gather(1, next_actions)
            else:
                next_target_q_values = self.target_network(next_states)
                next_q = next_target_q_values.max(dim=1, keepdim=True)[0]
            
            td_targets = rewards + (1 - dones) * self.gamma * next_q
        
        td_loss = F.mse_loss(current_q, td_targets)
        
        metrics = {
            "td_loss": td_loss.item(),
            "q_values_mean": current_q.mean().item(),
            "q_values_std": current_q.std().item(),
            "q_values_max": current_q.max().item(),
            "q_values_min": current_q.min().item(),
            "td_targets_mean": td_targets.mean().item(),
        }
        
        return td_loss, metrics
    
    def update(
        self,
        batch: Tuple,
    ) -> Dict[str, float]:
        """Perform a single gradient update step."""
        states, actions, rewards, next_states, dones = batch
        
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
        
        self.q_network.train()
        
        loss, metrics = self.compute_loss(
            states, actions, rewards, next_states, dones
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.grad_clip
            )
            metrics["grad_norm"] = grad_norm.item()
        
        self.optimizer.step()
        
        self._soft_update_target()
        
        self.training_step += 1
        metrics["training_step"] = self.training_step
        
        return metrics
    
    def _soft_update_target(self):
        """Soft update target network."""
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
        """Select action using greedy policy."""
        if isinstance(state, int):
            state = torch.tensor([state])
        elif isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        
        if state.dim() == 0:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        self.q_network.eval()
        
        with torch.no_grad():
            q_values = self.q_network(state)
            
            if admissible_actions is not None:
                mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
                mask[admissible_actions] = True
                q_values = q_values.masked_fill(~mask.unsqueeze(0), float('-inf'))
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def get_q_values(
        self,
        states: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Get Q-values for given states."""
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        
        states = states.to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(states)
        
        return q_values.cpu().numpy()
    
    def save(self, filepath: str):
        """Save model checkpoint."""
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
                "gamma": self.gamma,
                "tau": self.tau,
                "use_double_dqn": self.use_double_dqn,
                "use_one_hot": self.use_one_hot,
            },
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved DQN checkpoint to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
        
        logger.info(f"Loaded DQN checkpoint from {filepath}")
    
    @classmethod
    def from_checkpoint(cls, filepath: str, device: str = "auto") -> "DQN":
        """Create DQN agent from checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu")
        hp = checkpoint["hyperparameters"]
        
        agent = cls(
            state_dim=hp["state_dim"],
            action_dim=hp["action_dim"],
            hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            lr=hp["lr"],
            gamma=hp["gamma"],
            tau=hp["tau"],
            use_double_dqn=hp["use_double_dqn"],
            use_one_hot=hp["use_one_hot"],
            device=device,
        )
        agent.load(filepath)
        
        return agent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    agent = DQN(state_dim=716, action_dim=25, hidden_dim=256)
    
    batch_size = 32
    states = torch.randint(0, 716, (batch_size,))
    actions = torch.randint(0, 25, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randint(0, 716, (batch_size,))
    dones = torch.zeros(batch_size)
    
    batch = (states, actions, rewards, next_states, dones)
    
    metrics = agent.update(batch)
    print("Update metrics:", metrics)
    
    action = agent.select_action(100)
    print(f"Selected action: {action}")
    
    print("DQN test passed!")
