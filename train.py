import numpy as np
from jax import grad, jit, vmap, pmap
import time
import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import optax
from flax.training import train_state
from flax.training import checkpoints
import functools
from timeit import default_timer
from ns2d_turb_dm import ns2d_dm
from ns2d_dm import ns2d_dm as ns2d_dm_mat
from channel_dm import channel_dm
from argparse import ArgumentParser
import jax.tools.colab_tpu
import os
import dataclasses
from FNO import FNO2d
from UNO import UNO
from AELSTM import CAE_LSTM
from DilResNet import DilResNet
from UFNET import UFNET
from UFNET2 import UFNET as UFNET2
from u_vit2d import u_vit
from train_utils import train, create_state
import wandb
import yaml
  
def main(args):
    #os.environ['WANDB_API_KEY']='$37369cac70415978f6e8f2f17abb6feb280572a8'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
    #os.environ['WANDB_MODE']="offline"
    os.environ['XLA_FLAGS']='--xla_gpu_strict_conv_algorithm_picker={}'.format('false')
    if args.tpu:
      jax.tools.colab_tpu.setup_tpu()

    print(jax.devices())
    print(jax.local_device_count())

    
  
    if args.config:
      with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
      if config['id'] == False:
        id = config['model'] + '_' + config['dataset']  +'_'+  wandb.util.generate_id()
      else:
        id = config['id']
      wandb.init(project="turbulence_surrogates", entity="deepl4cfd", config=config, id=id, resume='allow', dir=config['save_path'])
    else:
      if not args.id:
        id = args.model + '_' + args.dataset  +'_'+  wandb.util.generate_id()
      else:
        id = args.id
      wandb.init(project="turbulence_surrogates", entity="deepl4cfd", config=args, id=id, resume='allow', dir=args.save_path)

    batch_size = wandb.config['batch_size']
    dataset = wandb.config['dataset']

    if dataset == '2dturb':
      dm = ns2d_dm(wandb.config.data_path,batch_size)
    if dataset == '2dturb_mat':
      dm = ns2d_dm_mat(wandb.config.data_path,batch_size)
    if dataset == 'channel':
      dm = channel_dm(wandb.config.data_path,batch_size)

    dm.setup()
    field = next(iter(dm.train_dataloader()))
    key =  jax.random.PRNGKey(3)

    if wandb.config.model == 'FNO':
      model =  FNO2d(output_features = wandb.config.output_features, T=4, modes=wandb.config['modes'])
    if wandb.config.model =='UNO':
      model = UNO(output_feats = wandb.config.output_features, modes = wandb.config.modes)
    if wandb.config.model == 'DilResNet':
      model = DilResNet(output_feats = wandb.config.output_features)
    if wandb.config.model == 'CAE_LSTM':
      model = CAE_LSTM(output_features = wandb.config.output_features)
    if wandb.config.model == 'UFNET':
      model = UFNET(output_feats = wandb.config.output_features)
    if wandb.config.model == 'UFNET2':
      model = UFNET2(output_feats = wandb.config.output_features)
    if wandb.config.model == 'UVIT':
      model = u_vit(base_channels = wandb.config.output_features)
    key, subkey =  jax.random.split(key,2)
    if wandb.config.model == 'CAE_LSTM':
      (_, carry), params  = model.init_with_output(subkey, field[0,:128,:128][...,:wandb.config.input_steps].numpy(), (None,None))
    elif wandb.config.model == 'UVIT':
      x = jnp.ones(field[0,:128,:128][...,:wandb.config.input_steps].numpy().shape, dtype=jnp.float16)
      params = model.init(subkey, x, False)
      carry = None
    else:
      params = model.init(subkey, field[0,:128,:128][...,:wandb.config.input_steps].numpy())
      carry = None
    del field

    alpha = wandb.config['lr']
    epoch_start = 0
    key, newkey = jax.random.split(key,2)
    state = create_state(params, alpha, model, checkpoint=wandb.config['checkpoint'], 
                        steps_per_epoch = len(dm.train_dataloader()), 
                        carry = carry, time_window=wandb.config.time_window,
                        epsilon = [wandb.config.epsilon_data, wandb.config.epsilon_eq],
                        dropout_key = newkey)

    train(state, dm)

if __name__ == "__main__":
  parser = ArgumentParser(add_help=False)
  parser = ArgumentParser(add_help=False)
  parser.add_argument('--max_epochs', type=int, default=100)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--pf_steps', type=int, default=2)
  parser.add_argument('--unroll_steps', type=int, default=4)
  parser.add_argument('--input_steps', type=int, default=1)
  parser.add_argument('--time_window', type=int, default=50)
  parser.add_argument('--output_features', type=int, default=1)
  parser.add_argument('--modes', type=int, default=12)
  parser.add_argument('--ds', type=int, default=2)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--data_path', type=str)
  parser.add_argument('--save_path', type=str, default = './')
  parser.add_argument('--model', type=str)
  parser.add_argument('--checkpoint', type=str, default=None)
  parser.add_argument('--train_loss', type=str, default=None)
  parser.add_argument('--x_data', type=float, default=1.0)
  parser.add_argument('--x_diff', type=float, default=1.0)
  parser.add_argument('--x_adv', type=float, default=1.0)
  parser.add_argument('--x_dt', type=float, default=1.0)
  parser.add_argument('--x_stab', type=float, default=1.0)
  parser.add_argument('--tpu', action='store_true')
  parser.add_argument('--push_forward', action='store_true')
  parser.add_argument('--denoise', action='store_true')
  parser.add_argument('--teacher_forcing', action='store_true')
  parser.add_argument('--unroll_loss', action='store_true')
  parser.add_argument('--dataset', type=str, default='2dturb')
  parser.add_argument('--id', type=str)
  parser.add_argument('--config', type=str, default=None)
  args = parser.parse_args()

  main(args)