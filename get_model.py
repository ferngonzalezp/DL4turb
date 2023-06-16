from UFNET2 import UFNET
from FNO import FNO2d
from UNO import UNO
import jax.numpy as jnp
from train_utils import create_state
import jax

def get_model(model, key,  input_shape, checkpoint, output_features, input_steps, **kwargs):
  nx = input_shape[0]
  ny = input_shape[1]
  x = jnp.ones(input_shape)[...,:input_steps]
  x = x.reshape(nx,ny,output_features,-1)
  if model == 'UNO':
    model_cls = UNO(output_feats=output_features, **kwargs)
  if model == 'FNO':
    model_cls = FNO2d(output_features=output_features, **kwargs)
  if model == 'UFNET':
    model_cls = UFNET(output_feats=output_features)
  
  params = model_cls.init(key, x)
  carry =  None

  epoch_start = 0
  key, newkey = jax.random.split(key,2)
  state = create_state(params, 1e-3, model_cls, checkpoint=checkpoint, 
                        steps_per_epoch = 100, 
                        carry = carry, time_window=10,
                        prefix = 'ckpt_epoch',
                        epsilon = [1, 1],
                        dropout_key = newkey)
  return state