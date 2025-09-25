"""Llama 4 Maverick MoE implementation with 17Bx128E architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .base_moe import BaseMoE, MoELayer
from ..config import ModelConfig


class LlamaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMaverickAttention(nn.Module):
    """Multi-head attention with grouped query attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Project to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class LlamaMaverickDecoderLayer(nn.Module):
    """Llama Maverick decoder layer with MoE."""
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_dim
        self.layer_idx = layer_idx
        
        # Self attention
        self.self_attn = LlamaMaverickAttention(config)
        
        # MoE layer instead of regular FFN
        self.mlp = MoELayer(
            hidden_size=config.hidden_dim,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            activation_fn="silu",
            router_aux_loss_coef=config.router_aux_loss_coef,
            router_z_loss_coef=config.router_z_loss_coef,
        )
        
        # Layer norms
        self.input_layernorm = LlamaRMSNorm(config.hidden_dim)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        routing_weights: Optional[torch.Tensor] = None,
        selected_experts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Dict[str, torch.Tensor]]:
        
        residual = hidden_states
        
        # Self attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MoE layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, moe_aux_losses = self.mlp(
            hidden_states,
            routing_weights=routing_weights,
            selected_experts=selected_experts
        )
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs + (moe_aux_losses,)


class LlamaMaverickMoE(BaseMoE):
    """Llama 4 Maverick MoE model with 17B parameters per expert and 128 experts."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.padding_idx = 0
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim, self.padding_idx)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            LlamaMaverickDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = LlamaRMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_moe_layers(self) -> List[MoELayer]:
        """Get all MoE layers in the model."""
        return [layer.mlp for layer in self.layers]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        routing_weights: Optional[List[torch.Tensor]] = None,
        selected_experts: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        seq_length_with_past = seq_length
        past_key_values_length = 0
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        
        hidden_states = inputs_embeds
        
        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_moe_aux_losses = []
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # Get routing info for this layer if provided
            layer_routing_weights = routing_weights[idx] if routing_weights is not None else None
            layer_selected_experts = selected_experts[idx] if selected_experts is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                routing_weights=layer_routing_weights,
                selected_experts=layer_selected_experts,
            )
            
            hidden_states = layer_outputs[0]
            moe_aux_losses = layer_outputs[-1]  # MoE auxiliary losses
            all_moe_aux_losses.append(moe_aux_losses)
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Aggregate MoE auxiliary losses
        total_aux_loss = sum(losses["aux_loss"] for losses in all_moe_aux_losses)
        total_z_loss = sum(losses["z_loss"] for losses in all_moe_aux_losses)
        
        outputs = {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "z_loss": total_z_loss,
            "moe_aux_losses": all_moe_aux_losses,
        }
        
        if past_key_values is not None:
            outputs["past_key_values"] = next_decoder_cache
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        if output_attentions:
            outputs["attentions"] = all_self_attns
        
        return outputs
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Prepare causal attention mask."""
        batch_size, seq_length = input_shape
        combined_attention_mask = None
        device = inputs_embeds.device
        
        if seq_length > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape, inputs_embeds.dtype, device=device, past_key_values_length=past_key_values_length
            )
        
        if attention_mask is not None:
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=seq_length).to(device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        
        return combined_attention_mask
    
    def _make_causal_mask(self, input_ids_shape, dtype, device, past_key_values_length=0):
        """Make causal attention mask."""
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
    def _expand_mask(self, mask, dtype, tgt_len=None):
        """Expand attention mask."""
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    
    def _compute_base_flops(self, input_length: int) -> float:
        """Compute FLOPs for non-MoE components."""
        # Embedding layer
        embed_flops = 0  # Embedding lookup is not counted as FLOPs
        
        # Attention layers
        attention_flops = 0
        for _ in range(self.num_layers):
            # Q, K, V projections
            qkv_flops = 3 * input_length * self.hidden_size * self.hidden_size
            
            # Attention computation
            attn_flops = input_length * input_length * self.hidden_size
            
            # Output projection
            out_flops = input_length * self.hidden_size * self.hidden_size
            
            attention_flops += qkv_flops + attn_flops + out_flops
        
        # Layer norms (approximated)
        norm_flops = 2 * self.num_layers * input_length * self.hidden_size
        
        # Output projection
        output_flops = input_length * self.hidden_size * self.vocab_size
        
        total_base_flops = embed_flops + attention_flops + norm_flops + output_flops
        return total_base_flops
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load pre-trained model (placeholder implementation)."""
        # This would normally load from HuggingFace or other model hub
        # For now, create a default configuration
        config = ModelConfig(
            model_type="llama_maverick",
            num_experts=128,
            expert_size=17_000_000_000,
            hidden_dim=8192,
            num_layers=80,
            vocab_size=128256,
            intermediate_size=28672,
            num_attention_heads=64,
            num_key_value_heads=8,
        )
        
        model = cls(config)
        # In a real implementation, you would load the pre-trained weights here
        return model

