"""
Behavior Cloning (BC) baseline for offline reinforcement learning.

Behavior Cloning learns a policy by supervised learning to imitate the
actions in the offline dataset. This serves as a simple baseline that
captures the behavior policy's performance.

This module provides:
- BehaviorCloning: Supervised learning policy that imitates dataset actions
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


class PolicyNetwork(nn.Module):
    """
    Policy network for behavior cloning.
    
    Outputs a probability distribution over discrete actions.
    
    Args:
        state_dim: Number of discrete states (for embedding) or state dimension
        action_dim: Number of discrete actions
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        use_one_hot: Whether to use one-hot state encoding
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
        self.use_one_hot = use_one_hot
        
        # Input dimension
        input_dim = state_dim if use_one_hot else hidden_dim
        
        # Embedding layer for discrete states
        if not use_one_hot:
            self.state_embedding = nn.Embedding(state_dim, hidden_dim)
        
        # Build MLP layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Output layer (logits)
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialise weights
        self._initialise_weights()
    
    def _initialise_weights(self):
        """Initialise network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits.
        
        Args:
            states: State tensor
        
        Returns:
            Action logits of shape (batch_size, action_dim)
        """
        if self.use_one_hot:
            x = states.float()
        else:
            states = states.long()
            x = self.state_embedding(states)
        
        return self.network(x)
    
    def get_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (softmax of logits)."""
        logits = self.forward(states)
        return F.softmax(logits, dim=-1)
    
    def get_action(
        self,
        states: torch.Tensor,
        admissible_actions: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Get action for given states.
        
        Args:
            states: State tensor
            admissible_actions: Optional mask of admissible actions
            deterministic: If True, return argmax; if False, sample
        
        Returns:
            Action indices
        """
        logits = self.forward(states)
        
        if admissible_actions is not None:
            logits = logits.masked_fill(~admissible_actions.bool(), float('-inf'))
        
        if deterministic:
            return logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)


class BehaviorCloning:
    """
    Behavior Cloning for offline reinforcement learning.
    
    BC learns a policy by supervised learning to maximize the likelihood
    of actions taken in the offline dataset:
    
        L_BC = -E_{(s,a)~D}[log π(a|s)]
    
    This is a simple baseline that captures the behavior policy's performance.
    It does not attempt to improve upon the behavior policy.
    
    Args:
        state_dim: Number of discrete states
        action_dim: Number of discrete actions
        hidden_dim: Network hidden dimension
        num_layers: Number of hidden layers
        lr: Learning rate
        weight_decay: L2 regularization coefficient
        use_one_hot: Whether to use one-hot state encoding
        device: Device to run on
    
    Example:
        >>> bc = BehaviorCloning(state_dim=716, action_dim=25)
        >>> metrics = bc.update(batch)
        >>> action = bc.select_action(state)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        use_one_hot: bool = False,
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
        self.weight_decay = weight_decay
        self.use_one_hot = use_one_hot
        
        # Initialise policy network
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_one_hot=use_one_hot,
        ).to(self.device)
        
        # Optimiser
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Training statistics
        self.training_step = 0
        
        logger.info(f"Initialised BehaviourCloning agent on {self.device}")
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute behavior cloning loss (negative log-likelihood).
        
        Args:
            states: Batch of states
            actions: Batch of actions taken in dataset
        
        Returns:
            loss: Cross-entropy loss
            metrics: Dictionary of logging metrics
        """
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device).long()
        
        # Get logits
        logits = self.policy(states)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, actions)
        
        # Compute accuracy
        predicted_actions = logits.argmax(dim=-1)
        accuracy = (predicted_actions == actions).float().mean()
        
        # Compute entropy (for monitoring)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        metrics = {
            "bc_loss": loss.item(),
            "accuracy": accuracy.item(),
            "entropy": entropy.item(),
        }
        
        return loss, metrics
    
    def update(
        self,
        batch: Tuple[
            Union[np.ndarray, torch.Tensor],
            Union[np.ndarray, torch.Tensor],
            ...
        ],
    ) -> Dict[str, float]:
        """
        Perform a single gradient update step.
        
        Args:
            batch: Tuple containing at least (states, actions, ...)
        
        Returns:
            metrics: Dictionary of training metrics
        """
        # Unpack batch (only need states and actions for BC)
        states, actions = batch[0], batch[1]
        
        # Convert to tensors
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        
        # Set to training mode
        self.policy.train()
        
        # Compute loss
        loss, metrics = self.compute_loss(states, actions)
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update step counter
        self.training_step += 1
        metrics["training_step"] = self.training_step
        
        return metrics
    
    def select_action(
        self,
        state: Union[int, np.ndarray, torch.Tensor],
        eval_mode: bool = True,
        admissible_actions: Optional[Union[List[int], np.ndarray]] = None,
        deterministic: bool = True,
    ) -> int:
        """
        Select action using learned policy.
        
        Args:
            state: Current state
            eval_mode: Whether in evaluation mode
            admissible_actions: Optional list of admissible action indices
            deterministic: If True, return argmax; if False, sample
        
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
        
        # Set to eval mode
        self.policy.eval()
        
        with torch.no_grad():
            logits = self.policy(state)
            
            if admissible_actions is not None:
                mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
                mask[admissible_actions] = True
                logits = logits.masked_fill(~mask.unsqueeze(0), float('-inf'))
            
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze().item()
        
        return action
    
    def get_action_probs(
        self,
        states: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Get action probabilities for given states.
        
        Args:
            states: States to evaluate
        
        Returns:
            Action probabilities array of shape (batch_size, action_dim)
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        
        states = states.to(self.device)
        
        self.policy.eval()
        with torch.no_grad():
            probs = self.policy.get_action_probs(states)
        
        return probs.cpu().numpy()
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "hyperparameters": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "use_one_hot": self.use_one_hot,
            },
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved BC checkpoint to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
        
        logger.info(f"Loaded BC checkpoint from {filepath}")
    
    @classmethod
    def from_checkpoint(cls, filepath: str, device: str = "auto") -> "BehaviorCloning":
        """Create BC agent from checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu")
        hp = checkpoint["hyperparameters"]
        
        agent = cls(
            state_dim=hp["state_dim"],
            action_dim=hp["action_dim"],
            hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
            use_one_hot=hp["use_one_hot"],
            device=device,
        )
        agent.load(filepath)
        
        return agent


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    # Create BC agent
    agent = BehaviorCloning(
        state_dim=716,
        action_dim=25,
        hidden_dim=256,
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
    
    print("BC test passed!")
