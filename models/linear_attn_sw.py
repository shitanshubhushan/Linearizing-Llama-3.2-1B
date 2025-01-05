import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache

class HybridAttention(LlamaSdpaAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.window_size = 64
        #self.layer_idx = layer_idx

        # Initialize learnable factors
        # Create one factor pair per attention head
        num_heads = config.num_attention_heads
        self.window_factors = torch.nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)
        self.linear_factors = torch.nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)

        # Optional: Add sigmoid to ensure factors stay positive
        self.factor_activation = torch.nn.Sigmoid()

    def sliding_window_attention(self, query_states, key_states, value_states, window_size, window_factor):
        """Compute sliding window attention"""
        batch_size, num_heads, seq_len, head_dim = query_states.shape

        key_windows = F.pad(key_states, (0, 0, window_size - 1, 0), value=0)
        key_windows = key_windows.unfold(2, window_size, 1)

        value_windows = F.pad(value_states, (0, 0, window_size - 1, 0), value=0)
        value_windows = value_windows.unfold(2, window_size, 1)

        attn_weights = torch.einsum('bhld,bhldw->bhlw', query_states, key_windows) * (head_dim ** -0.5)
        attn_weights = torch.where(attn_weights == 0,
                                 torch.tensor(-float('inf'), device=attn_weights.device),
                                 attn_weights)

        # Apply learnable window factor (with sigmoid to ensure positivity)
        attn_weights = self.factor_activation(window_factor) * F.softmax(attn_weights, dim=-1)

        attn_output = torch.einsum('bhlw,bhldw->bhld', attn_weights, value_windows)
        sum_weights = attn_weights.sum(dim=-1, keepdim=True)

        return attn_output, sum_weights

    def linear_attention(self, query_states, key_states, value_states, window_size, linear_factor):
        """Compute linear attention with cumsum"""
        def feature_map(x):
            return F.elu(x) + 1

        query_prime = feature_map(query_states)
        key_prime = feature_map(key_states)

        key_prime = F.pad(key_prime, (0, 0, window_size, 0), value=0)[:, :, :-window_size, :]
        value_padded = F.pad(value_states, (0, 0, window_size, 0), value=0)[:, :, :-window_size, :]

        # Compute KV
        kv = torch.einsum('bhlf,bhld->bhlfd', key_prime, value_padded)
        # Apply learnable linear factor (with sigmoid to ensure positivity)
        qkv = self.factor_activation(linear_factor) * torch.einsum('bhlf,bhlfd->bhld',
                                                                  query_prime,
                                                                  kv.cumsum(dim=2))

        sum_k = key_prime.cumsum(dim=2)
        sum_qk = self.factor_activation(linear_factor) * torch.einsum('bhld,bhld->bhl',
                                                                     query_prime,
                                                                     sum_k)[..., None]
        sum_qk = torch.where(sum_qk == 0, torch.tensor(1e-12, device=sum_qk.device), sum_qk)

        return qkv, sum_qk

    def hybrid_attention(self, query_states, key_states, value_states):
        """Combine sliding window and linear attention with learnable factors"""
        qkv_window, sum_window = self.sliding_window_attention(
            query_states, key_states, value_states,
            self.window_size, self.window_factors
        )

        qkv_linear, sum_linear = self.linear_attention(
            query_states, key_states, value_states,
            self.window_size, self.linear_factors
        )

        output = (qkv_window + qkv_linear) / (sum_window + sum_linear)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self.hybrid_attention(
            query_states,
            key_states,
            value_states
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value