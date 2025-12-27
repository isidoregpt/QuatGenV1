"""
Policy Network for Molecular Generation
Transformer-based autoregressive model with RL fine-tuning

Architecture:
- GPT-2 style decoder-only transformer
- Generates SMILES tokens autoregressively
- Fine-tuned with policy gradient methods (REINFORCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Tuple, List
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class MoleculePolicy(nn.Module):
    """
    Policy network for molecular generation
    
    Generates SMILES tokens autoregressively using a transformer decoder.
    Can be trained with:
    - Maximum likelihood (teacher forcing)
    - Policy gradient (REINFORCE with baseline)
    - PPO (Proximal Policy Optimization)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        # Value head for REINFORCE baseline
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Padding mask [batch, seq_len]
        
        Returns:
            logits: Token logits [batch, seq_len, vocab_size]
            values: State values [batch, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Causal mask
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)
        
        # Output
        x = self.out_norm(x)
        logits = self.out_proj(x)
        values = self.value_head(x).squeeze(-1)
        
        return logits, values
    
    def get_action_prob(
        self,
        input_ids: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and entropy for given actions
        
        Used in PPO training
        """
        logits, values = self.forward(input_ids)
        
        # Get distribution
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # Log probs of taken actions
        log_probs = dist.log_prob(actions)
        
        # Entropy for exploration bonus
        entropy = dist.entropy()
        
        return log_probs, entropy, values
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate molecules autoregressively
        
        Args:
            batch_size: Number of molecules to generate
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
            max_length: Maximum sequence length
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            device: Device to generate on
        
        Returns:
            sequences: Generated token IDs [batch, seq_len]
            log_probs: Log probabilities [batch, seq_len]
        """
        if max_length is None:
            max_length = self.max_len
        
        if device is None:
            device = next(self.parameters()).device
        
        # Start with BOS token
        sequences = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        log_probs_list = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Forward pass
            logits, _ = self.forward(sequences)
            next_logits = logits[:, -1, :]  # [batch, vocab]
            
            # Apply temperature
            next_logits = next_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][:, -1:]
                next_logits[indices_to_remove] = float("-inf")
            
            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            dist = Categorical(probs)
            next_tokens = dist.sample()
            next_log_probs = dist.log_prob(next_tokens)
            
            # Mask finished sequences
            next_tokens = torch.where(finished, eos_token_id, next_tokens)
            next_log_probs = torch.where(finished, torch.zeros_like(next_log_probs), next_log_probs)
            
            # Append
            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            log_probs_list.append(next_log_probs)
            
            # Check for EOS
            finished = finished | (next_tokens == eos_token_id)
            if finished.all():
                break
        
        log_probs = torch.stack(log_probs_list, dim=1)
        
        return sequences, log_probs
    
    def compute_loss_ml(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute maximum likelihood loss (teacher forcing)
        
        Used for pre-training
        """
        logits, _ = self.forward(input_ids, attention_mask)
        
        # Shift for autoregressive prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Cross entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.pad_token_id
        )
        
        return loss
    
    def compute_loss_rl(
        self,
        sequences: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute REINFORCE loss with baseline
        
        Args:
            sequences: Generated sequences [batch, seq_len]
            log_probs: Log probabilities of actions [batch, seq_len-1]
            rewards: Rewards for each sequence [batch]
            values: Value estimates [batch, seq_len]
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
        
        Returns:
            policy_loss: Policy gradient loss
            value_loss: Value function loss
            metrics: Dictionary of training metrics
        """
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        # Expand rewards to sequence level (reward at EOS, 0 elsewhere)
        sequence_rewards = torch.zeros(batch_size, seq_len - 1, device=device)
        sequence_rewards[:, -1] = rewards
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(sequence_rewards)
        last_gae = 0
        
        for t in reversed(range(seq_len - 1)):
            if t == seq_len - 2:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = sequence_rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * gae_lambda * last_gae
        
        # Policy loss (negative because we maximize)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        returns = advantages + values[:, :-1]
        value_loss = F.mse_loss(values[:, :-1], returns.detach())
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item()
        }
        
        return policy_loss, value_loss, metrics


class PolicyOptimizer:
    """
    Optimizer for policy network
    
    Handles learning rate scheduling and gradient clipping
    """
    
    def __init__(
        self,
        policy: MoleculePolicy,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000
    ):
        self.policy = policy
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,
            T_mult=2
        )
    
    def step(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        policy_weight: float = 1.0,
        value_weight: float = 0.5
    ):
        """Perform optimization step"""
        self.optimizer.zero_grad()
        
        # Combined loss
        loss = policy_weight * policy_loss + value_weight * value_loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.max_grad_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1
        
        return loss.item()
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
    
    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step_count": self.step_count
        }
    
    def load_state_dict(self, state_dict: dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.step_count = state_dict["step_count"]
