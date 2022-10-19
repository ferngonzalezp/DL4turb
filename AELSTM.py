import jax
from jax import numpy as jnp
from flax import linen as nn
from jax import lax
import math

class res_block(nn.Module):

  out_feats: int
  kernel: tuple

  @nn.compact
  def __call__(self, x):
    x1 = nn.Conv(self.out_feats, self.kernel)(x)
    x1 = nn.gelu(x1)
    x1 = nn.Conv(self.out_feats, self.kernel)(x1)
    x2 = nn.Conv(self.out_feats, self.kernel)(x)
    return x1 + x2

class Encoder(nn.Module):

  @nn.compact
  def __call__(self,x):
    out_feats = 16
    x = x.reshape(x.shape[0], x.shape[1],-1)
    x1 = res_block(out_feats, [3,3])(x)
    x1 = nn.gelu(x1)
    x2 = res_block(out_feats, [5,5])(x)
    x2 = nn.gelu(x2)
    x3 = res_block(out_feats, [7,7])(x)
    x3 = nn.gelu(x3)

    for i in range(5):
      if i%2 == 0:
        out_feats = math.floor(out_feats*3/2)

      x1 = res_block(out_feats, [3,3])(x1)
      x1 = nn.gelu(x1)
      x1 = nn.max_pool(x1,(2,2),(2,2))
      x2 = res_block(out_feats, [5,5])(x2)
      x2 = nn.gelu(x2)
      x2 = nn.max_pool(x2,(2,2),(2,2))
      x3 = res_block(out_feats, [7,7])(x3)
      x3 = nn.gelu(x3)
      x3 = nn.max_pool(x3,(2,2),(2,2))
    
    return nn.gelu(x1 + x2 + x3)

class trans_res_block(nn.Module):

  out_feats: int
  kernel: tuple

  @nn.compact
  def __call__(self, x):
    x1 = nn.Conv(self.out_feats, self.kernel)(x)
    x1 = nn.gelu(x1)
    x1 = nn.ConvTranspose(self.out_feats, self.kernel,[2,2],2*[self.kernel[0]//2])(x1)
    x2 = nn.ConvTranspose(self.out_feats, self.kernel,[2,2],2*[self.kernel[0]//2])(x)
    return x1 + x2

class Decoder(nn.Module):

  output_features: int = 1

  def get_grid(self,shape):
        size_x, size_y = shape[0], shape[1]
        gridx, gridy = jnp.meshgrid(jnp.linspace(0, 1, size_x), jnp.linspace(0, 1, size_y))
        gridx = gridx.reshape(size_x, size_y, 1)
        gridy = gridy.reshape(size_x, size_y, 1)
        return jnp.concatenate((gridx, gridy), axis=-1)

  @nn.compact
  def __call__(self,x):
    out_feats = x.shape[-1]
    x = x.reshape(x.shape[0], x.shape[1],-1)
    x1 = trans_res_block(out_feats, [4,4])(x)
    #x1 = nn.GroupNorm(out_feats)(x1)
    x1 = nn.gelu(x1)
    x2 = trans_res_block(out_feats, [6,6])(x)
    #x2 = nn.GroupNorm(out_feats)(x2)
    x2 = nn.gelu(x2)
    x3 = trans_res_block(out_feats, [8,8])(x)
    #x3 = nn.GroupNorm(out_feats)(x3)
    x3 = nn.gelu(x3)

    for i in range(3):
      if i%2 == 0:
        out_feats = math.floor(out_feats*2/3)

      x1 = trans_res_block(out_feats, [4,4])(x1)
      #x1 = nn.GroupNorm(out_feats)(x1)
      x1 = nn.gelu(x1)
      x2 = trans_res_block(out_feats, [6,6])(x2)
      #x2 = nn.GroupNorm(out_feats)(x2)
      x2 = nn.gelu(x2)
      x3 = trans_res_block(out_feats, [8,8])(x3)
      #x3 = nn.GroupNorm(out_feats)(x3)
      x3 = nn.gelu(x3)

    x = x1 + x2 + x3
    nx = x.shape[0]
    ny = x.shape[1]
    coords = jnp.meshgrid(jnp.linspace(0,nx,math.floor(nx*2)), 
                          jnp.linspace(0,ny,math.floor(ny*2)))
    x = jax.vmap(jax.scipy.ndimage.map_coordinates, in_axes=(-1,None,None),out_axes=-1)(x,coords,1)
    x = nn.Conv(self.output_features,[3,3])(x)
    return x
    
class CAE_LSTM(nn.Module):

  output_features: int = 1
  lstm_cells: int = 4
  lstm_features: int = 128

  def setup(self):

    self.encoder = Encoder()
    self.decoder = Decoder(self.output_features)

    lstm_cells = []
    linear = []
    self.lstm_input_layer = nn.Conv(self.lstm_features, [3,3])
    for i in range(self.lstm_cells):
      lstm_cells.append(nn.ConvLSTM(self.lstm_features,[3,3]))
      linear.append(nn.Conv(self.lstm_features,[1,1]))
    
    self.lstm = lstm_cells
    self.linear = linear
    self.output_conv = nn.Conv(54, [3,3])
    
  def __call__(self,x, carry):
    nx = x.shape[0]
    ny = x.shape[1]
    x_input = x[...,-1:].reshape(nx,ny,-1)
    x = self.encoder(x)
    encoded_shape = x.shape
    x = self.lstm_input_layer(x)
    x = nn.gelu(x)
    shape = x.shape
    if None in carry:
      carry = self.lstm[0].initialize_carry(jax.random.PRNGKey(3),(1,),x.shape)
    x = x.reshape(1,*shape)
    for i in range(self.lstm_cells):
      carry, x = self.lstm[i](carry, x)
      x = self.linear[i](x)
      x = nn.gelu(x)
    x = self.output_conv(x)
    x = nn.gelu(x).reshape(encoded_shape)
    x = self.decoder(x)
    return x, carry