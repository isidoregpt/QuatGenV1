"""Policy Network for Molecular Generation - Transformer-based autoregressive model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model), nn.Dropout(dropout))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class MoleculePolicy(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
                 d_ff: int = 1024, max_len: int = 128, dropout: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz), diagonal=1).masked_fill(torch.triu(torch.ones(sz, sz), diagonal=1) == 1, float("-inf"))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pos_encoding(self.token_embedding(input_ids))
        mask = self._generate_causal_mask(input_ids.size(1)).to(input_ids.device)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.out_norm(x)
        return self.out_proj(x), self.value_head(x).squeeze(-1)
    
    @torch.no_grad()
    def generate(self, batch_size: int = 1, bos_token_id: int = 2, eos_token_id: int = 3,
                 max_length: Optional[int] = None, temperature: float = 1.0, top_k: int = 0,
                 top_p: float = 1.0, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        max_length = max_length or self.max_len
        device = device or next(self.parameters()).device
        sequences = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        log_probs_list = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            logits, _ = self.forward(sequences)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            dist = Categorical(probs)
            next_tokens = dist.sample()
            next_log_probs = dist.log_prob(next_tokens)
            next_tokens = torch.where(finished, eos_token_id, next_tokens)
            next_log_probs = torch.where(finished, torch.zeros_like(next_log_probs), next_log_probs)
            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            log_probs_list.append(next_log_probs)
            finished = finished | (next_tokens == eos_token_id)
            if finished.all():
                break
        return sequences, torch.stack(log_probs_list, dim=1)
    
    def compute_loss_rl(self, sequences: torch.Tensor, log_probs: torch.Tensor, rewards: torch.Tensor,
                        values: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        batch_size, seq_len = sequences.shape
        device = sequences.device
        sequence_rewards = torch.zeros(batch_size, seq_len - 1, device=device)
        sequence_rewards[:, -1] = rewards
        advantages = torch.zeros_like(sequence_rewards)
        last_gae = 0
        for t in reversed(range(seq_len - 1)):
            next_value = 0 if t == seq_len - 2 else values[:, t + 1]
            delta = sequence_rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * gae_lambda * last_gae
        policy_loss = -(log_probs * advantages.detach()).mean()
        returns = advantages + values[:, :-1]
        value_loss = F.mse_loss(values[:, :-1], returns.detach())
        return policy_loss, value_loss, {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "mean_reward": rewards.mean().item()}


class PolicyOptimizer:
    def __init__(self, policy: MoleculePolicy, lr: float = 1e-4, weight_decay: float = 0.01, max_grad_norm: float = 1.0):
        self.policy = policy
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10000, T_mult=2)
    
    def step(self, policy_loss: torch.Tensor, value_loss: torch.Tensor, policy_weight: float = 1.0, value_weight: float = 0.5):
        self.optimizer.zero_grad()
        loss = policy_weight * policy_loss + value_weight * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()
