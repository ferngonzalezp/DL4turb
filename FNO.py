



import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import numpy as np
import os

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
    x = jnp.fft.irfftn(out_ft, s=(x.shape[0], x.shape[1]), axes=(0,1))
    return x

class FNO2d(nn.Module):
    modes: int = 20
    width: int = 32
    T: int = 4
    output_features: int = 1
    def setup(self):
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Dense(self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        conv = []
        w = []

        for i in range(self.T):
          conv.append(SpectralConv2d_fast(self.width, self.width, self.modes, self.modes))
          w.append(nn.Conv(self.width, [1]))

        self.conv = conv
        self.w = w

        self.fc1 = nn.Dense(128)
        self.fc2 = nn.Dense(self.output_features)

    def __call__(self, x):
        size_x, size_y = [x.shape[0], x.shape[1]]
        x = x.reshape(size_x, size_y, -1)
        grid = self.get_grid(x.shape)
        x = jnp.concatenate((x, grid), axis=-1)
        x = self.fc0(x)
        x = x
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        for i in range(self.T):
          x1 = self.conv[i](x)
          x2 = self.w[i](x.reshape(-1,self.width)).reshape(size_x,size_y,-1)
          x = (x1 + x2)
          x = nn.swish(x)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = nn.swish(x)
        x = self.fc2(x)
        return x

    def get_grid(self,shape):
        size_x, size_y = shape[0], shape[1]
        gridx, gridy = jnp.meshgrid(jnp.linspace(0, 1, size_x), jnp.linspace(0, 1, size_y))
        gridx = gridx.reshape(size_x, size_y, 1)
        gridy = gridy.reshape(size_x, size_y, 1)
        return jnp.concatenate((gridx, gridy), axis=-1)