import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import numpy as np
import os
import math

def uniform_complex(scale=1e-2):
  def init(key, shape):
        key1, key2 = random.split(key,2)
        x = random.uniform(key1, shape)
        y = random.uniform(key2, shape)
        return jax.lax.complex(x,y)
  return init

class SpectralConv2d_fast(nn.Module):
  in_channels: int = 12
  out_channels: int = 1
  modes1: int = 12
  modes2: int = 12
  scale: float = 1/(in_channels*out_channels)
  init: Callable =  nn.initializers.uniform(scale)
  scale: float = 1

  @nn.compact
  def __call__(self,x):
    compl_mul2d =  lambda inputs, weights: jnp.einsum("xyi, ixyo -> xyo", inputs, weights)
    weights1 = self.param('weights1',self.init,(self.in_channels, self.modes1, self.modes2, self.out_channels,2))
    weights2 = self.param('weights2',self.init,(self.in_channels, self.modes1, self.modes2, self.out_channels,2))

    #Compute Fourier coeffcients up to factor of e^(- something constant)
    x_ft = jnp.fft.rfftn(x,axes=(0,1))
    # Multiply relevant Fourier modes
    out_ft = jnp.zeros((x.shape[0], x.shape[1]//2 + 1, self.out_channels,2))
    out_ft = out_ft.at[:self.modes1, :self.modes2, :,0].set(compl_mul2d(x_ft[:self.modes1, :self.modes2, :].real, weights1[...,0]))
    out_ft = out_ft.at[:self.modes1, :self.modes2, :,1].set(compl_mul2d(x_ft[:self.modes1, :self.modes2, :].imag, weights1[...,1]))
    out_ft = out_ft.at[-self.modes1:, :self.modes2, :, 0].set(compl_mul2d(x_ft[-self.modes1:, :self.modes2, :].real, weights2[...,0]))
    out_ft = out_ft.at[-self.modes1:, :self.modes2, :, 1].set(compl_mul2d(x_ft[-self.modes1:, :self.modes2, :].imag, weights2[...,1]))
    out_ft = jax.lax.complex(out_ft[...,0],out_ft[...,1])
    #Return to physical space
    x = jnp.fft.irfftn(out_ft, s=(math.floor(self.scale * x.shape[0]), math.floor(self.scale * x.shape[1])), axes=(0,1))
    return x

class operatorBlock(nn.Module):
  in_feats: int
  out_feats: int
  modes: int
  scale: float

  @nn.compact
  def __call__(self, x):
    nx = x.shape[0]
    ny = x.shape[1]
    x1 = SpectralConv2d_fast(self.in_feats, self.out_feats, self.modes, self.modes,
                             scale=self.scale)(x)
    x2 = nn.Conv(self.out_feats, [1,1])(x)
    coords = jnp.meshgrid(jnp.linspace(0,nx,math.floor(nx*self.scale)), 
                          jnp.linspace(0,ny,math.floor(ny*self.scale)))
    x2 = jax.vmap(jax.scipy.ndimage.map_coordinates, in_axes=(-1,None,None),out_axes=-1)(x2,coords,1)
    x = x1 + x2
    x = nn.GroupNorm()(x)
    x = nn.gelu(x)
    return x
  
class ds_conv(nn.Module):
    out_feats: int
    kernel: int
    strides: int

    @nn.compact
    def __call__(self, x):
      nx = x.shape[0]
      ny = x.shape[1]
      coords = jnp.meshgrid(jnp.linspace(0,nx,x.shape[0]//2), 
                          jnp.linspace(0,ny,x.shape[1]//2))
      x = jax.vmap(jax.scipy.ndimage.map_coordinates, in_axes=(-1,None,None),out_axes=-1)(x,coords,1)
      x1 = nn.Conv(self.out_feats, [self.kernel, self.kernel])(x)
      x2 = nn.Conv(self.out_feats, [1,1])(x)
      x = x1 + x2
      x = nn.GroupNorm()(x)
      x = nn.gelu(x)
      return x

class us_conv(nn.Module):
    out_feats: int
    kernel: int
    strides: int

    @nn.compact
    def __call__(self, x):
      nx = x.shape[0]
      ny = x.shape[1]
      x1 = nn.Conv(self.out_feats, [self.kernel, self.kernel])(x)
      x2 = nn.Conv(self.out_feats, [1,1])(x)
      x = x1 + x2
      coords = jnp.meshgrid(jnp.linspace(0,nx,int(x.shape[0]*2)), 
                          jnp.linspace(0,ny,int(x.shape[1]*2)))
      x = jax.vmap(jax.scipy.ndimage.map_coordinates, in_axes=(-1,None,None),out_axes=-1)(x,coords,1)
      x = nn.GroupNorm()(x)
      x = nn.gelu(x)
      return x

class UFNET(nn.Module):
  encoder_blocks: int = 3
  proc_layers: int = 3
  decoder_blocks: int = 3
  width_fc: int = 32
  output_feats: int = 1

  def setup(self):

    self.fc0 = nn.Dense(self.width_fc)

    encoder = []
    processor = []
    decoder = []
    width = self.width_fc
    for i in range(self.encoder_blocks):
      encoder.append(ds_conv(
                            width*2,
                            kernel = 4, strides = 1))
      width = width * 2
    for i in range(self.encoder_blocks):
      processor.append(operatorBlock(width, 
                    width,
                    modes = 8, scale = 1))
    for i in range(self.decoder_blocks):
      decoder.append(us_conv(
                            width//2,
                            kernel = 4, strides = 1))
      width = width//2
    self.encoder = encoder
    self.processor = processor
    self.decoder = decoder
    self.fc1 = nn.Dense(width)
    self.fc2 = nn.Dense(self.output_feats)
  
  def get_grid(self,shape):
        size_x, size_y = shape[0], shape[1]
        gridx, gridy = jnp.meshgrid(jnp.linspace(0, 1, size_x), jnp.linspace(0, 1, size_y))
        gridx = gridx.reshape(size_x, size_y, 1)
        gridy = gridy.reshape(size_x, size_y, 1)
        return jnp.concatenate((gridx, gridy), axis=-1)

  def __call__(self, x):

      size_x, size_y = [x.shape[0], x.shape[1]]
      x = x.reshape(size_x, size_y, -1)
      grid = self.get_grid(x.shape)
      x = jnp.concatenate((x, grid), axis=-1)
      x = self.fc0(x)
      h = []
      for i in range(self.encoder_blocks):
        x = self.encoder[i](x)
        h.append(x)
      
      for i in range(self.proc_layers):
        x = self.processor[i](x)
      
      for i in range(self.decoder_blocks):
        x = self.decoder[i](jnp.concatenate([x,h[-(i+1)]],-1))
      
      x = self.fc1(x)
      x = nn.gelu(x)
      x = self.fc2(x)

      return x    