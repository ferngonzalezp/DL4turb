import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import numpy as np
import os
import math
from einops import rearrange, repeat

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1.
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, ((0,0),(1,1)))
    return emb[0]

# MLP block

class mlp_block(nn.Module):
  expansion_factor: int
  dropout: float

  @nn.compact
  def __call__(self, x, deterministic):
    XYZ, C = x.shape
    #x = nn.LayerNorm()(x)
    mlp_h = nn.Dense(self.expansion_factor * C)(x)

    mlp_h = nn.swish(mlp_h)

    # add in timestep embedding
    #time_emb = nn.Dense(features=2 * self.expansion_factor * C)(nn.swish(time_emb))
    #time_emb = time_emb[:,  jnp.newaxis, jnp.newaxis, :]  # [B, H, W, C]
    #scale, shift = jnp.split(time_emb, 2, axis=-1)
    #mlp_h = mlp_h * (1 + scale) + shift

    mlp_h = nn.Dropout(self.dropout)(mlp_h, deterministic=deterministic)
    out = nn.Dense(C, kernel_init=nn.initializers.constant(0.0))(mlp_h)
    return out

def fixed_pos_embedding(x, seq_dim=0):
    dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))

    sinusoid_inp = np.einsum("i , j -> i j", np.arange(x.shape[seq_dim]), inv_freq)

    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, "... d j -> ... (d j)")

def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2)[:, None, :], sincos)
    return (x * cos) + (rotate_every_two(x) * sin)

# Galerkin attention block

class galerkin_attention(nn.Module):
  num_heads : int

  @nn.compact
  def __call__(self, x):
    x_norm = nn.LayerNorm()(x)

    seq_len, C = x.shape
    head_dim = C//self.num_heads

    qw = nn.DenseGeneral((self.num_heads, head_dim))(x)
    kw = nn.DenseGeneral((self.num_heads, head_dim))(x)
    vw = nn.DenseGeneral((self.num_heads, head_dim))(x)

    #apply RoPe
    sincos = fixed_pos_embedding(kw)
    kw = apply_rotary_pos_emb(kw, sincos)
    vw = apply_rotary_pos_emb(vw, sincos)

    #Attention

    kw = nn.LayerNorm(use_bias=True)(kw)
    vw = nn.LayerNorm(use_bias=True)(vw)
    weights = jnp.einsum('khi, khj -> ihj', kw, vw)
    attn_vals = jnp.einsum('qhi, ihd -> qhd', qw, weights) / seq_len
    
    out = nn.DenseGeneral(C, axis=(-2,-1), kernel_init = nn.initializers.constant(0.0))(attn_vals)
    return out

#transformer block

class transformer_block(nn.Module):
  num_heads : int
  expansion_factor: int
  dropout: float = 0.0

  @nn.compact
  def __call__(self,x, deterministic):
    x += mlp_block(self.expansion_factor, self.dropout)(x, deterministic)
    x += galerkin_attention(self.num_heads)(x)
    return x

# ResBlock

class resnet_block_ds(nn.Module):
  out_feats: int

  @nn.compact
  def __call__ (self, x, skip_h=None):
    X, Y, C = x.shape
    h = nn.LayerNorm(use_bias=True)(x)
    if skip_h is not None:
      skip_h = nn.LayerNorm(use_bias=True)(skip_h)
      h = (h + skip_h) / jnp.sqrt(2)
    h = nn.swish(h)
    h = nn.Conv(self.out_feats, [3,3])(h)
    h = nn.LayerNorm(use_bias=True)(h)

    # add in timestep embedding
    #time_emb = nn.Dense(features=2 * self.out_feats)(nn.swish(time_emb))
    #time_emb = time_emb[jnp.newaxis, jnp.newaxis, :]  # [B, X, Y, Z, C]
    #scale, shift = jnp.split(time_emb, 2, axis=-1)
    #h = h * (1 + scale) + shift

    #h = nn.swish(h)

    h = nn.swish(h)
    h = nn.Conv(self.out_feats, [3,3], kernel_init=nn.initializers.constant(0.0))(h)
    return x + h

class u_vit(nn.Module):

  num_res_blocks = [2, 2, 2]
  num_transformers_blocks: int = 36
  num_heads: int = 4
  dropout: float = 0.2
  expansion_factor: int = 4
  channel_multiplier = [2, 4, 16]
  base_channels: int = 1
  time_embedding_dim: int = 1024
  dtype: Any = jnp.float16

  @nn.compact
  def __call__ (self, x, training: bool = False):
    X, Y = [x.shape[0], x.shape[1]]
    x = x.reshape(X, Y, -1)

    #t_emb = get_timestep_embedding(time, self.time_embedding_dim)

    # use sinusoidal embeddings to encode timesteps
    #t_emb = nn.Dense(features=self.base_channels * 4)(t_emb)
    #t_emb = nn.Dense(features=self.base_channels * 4,)(nn.swish(t_emb))  

    h0 = nn.Conv(self.base_channels, [1,1])(x)
    hs = []

    #dx_emb = nn.Conv(self.base_channels, [1,1,1])(dx)
    
    #h0 = jnp.concatenate([h0, dx_emb], axis=-1)
    last_h = h0

    last_channel = last_h.shape[-1]

    #down path
    for i_level in range(len(self.num_res_blocks)):
      for i_block in range(self.num_res_blocks[i_level]):
        last_h = resnet_block_ds(last_channel)(last_h)
        hs.append(last_h)
      
      last_h = nn.Conv(last_channel * self.channel_multiplier[i_level], [1,1])(last_h)
      C = last_channel = last_h.shape[-1]
      last_h = nn.avg_pool(last_h, (2,2), (2,2))
    
    #Transformer
    X, Y, C = last_h.shape
    last_h = last_h.reshape((X*Y,C))
    for i in range(self.num_transformers_blocks):
      last_h = transformer_block(self.num_heads, self.expansion_factor, self.dropout)(last_h, deterministic = not training)
    last_h = last_h.reshape((X,Y,C))

    #Up_path
    c = -1
    for i_level in range(len(self.num_res_blocks)):
      last_h = nn.ConvTranspose(C // self.channel_multiplier[-1-i_level], [3,3], [2,2])(last_h)
      C = last_h.shape[-1]
      for i_block in range(self.num_res_blocks[i_level]):
        last_h = resnet_block_ds(C)(last_h, hs[c])
        c = c - 1
    
    out = nn.Dense(self.base_channels)(last_h)
    return out