import jax
from jax import numpy as jnp
from flax import linen as nn
from jax import lax
import math

class DilResBlock(nn.Module):
  dilation: int = 1
  N_layers : int  = 4

  @nn.compact
  def __call__(self,x):
    x1 = nn.Conv(48,[1,1])(x)
    for i in range(self.N_layers):
      x = nn.Conv(48,[3,3], kernel_dilation = self.dilation)(x)
      x = nn.relu(x)
    return x + x1
    
class DilResNet(nn.Module):
  output_feats: int = 1
  ed_dilations = [1,2,4]
  proc_dilations = [8]

  def setup(self):
    proc_encoder = []
    proc_processor = []
    proc_decoder = []
    self.encoder = nn.Conv(48,[3,3])
    self.decoder = nn.Conv(self.output_feats,[3,3])

    for i in range(len(self.ed_dilations)):
      proc_encoder.append(DilResBlock(self.ed_dilations[i]))
      proc_decoder.append(DilResBlock(self.ed_dilations[-i-1]))
    
    for i in range(len(self.proc_dilations)):
      proc_processor.append(DilResBlock(self.proc_dilations[i]))

    self.proc_encoder = proc_encoder
    self.proc_processor = proc_processor
    self.proc_decoder = proc_decoder

  def __call__(self,x):
    nx = x.shape[0]
    ny = x.shape[1]
    x_input = x[...,-1:].reshape(nx,ny,-1)
    x = x.reshape(nx,ny,-1)
    x = self.encoder(x)
    h = []

    for i in range(len(self.ed_dilations)):
      x = self.proc_encoder[i](x)
      x = nn.relu(x)
      h.append(x)
    
    for i in range(len(self.proc_dilations)):
      x = self.proc_processor[i](x)
      x = nn.relu(x)

    for i in range(len(self.ed_dilations)):
      x = self.proc_decoder[i](jnp.concatenate([x,h[-1-i]],axis=-1))
      x = nn.relu(x)

    x = self.decoder(x)

    return x + x_input