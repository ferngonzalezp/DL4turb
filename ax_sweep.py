from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting
from train_utils import train
import wandb
import torch
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
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
from train_utils import train, create_state
import yaml
from argparse import ArgumentParser


def main(args):

  os.environ['WANDB_API_KEY']='$37369cac70415978f6e8f2f17abb6feb280572a8'
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
  os.environ['WANDB_MODE']="offline"
  os.environ['WANDB_DIR']=args.save_path
  os.environ['XLA_FLAGS']='--xla_gpu_strict_conv_algorithm_picker={}'.format('false')

  if args.checkpoint:
    ax_client = AxClient.load_from_json_file(filepath=args.save_path+'/'+args.name+'.json')
  else:
    ax_client = AxClient()

  parameters=[
          {
              "name": "lr",
              "type": "range",
              "bounds": [1e-7, 1e-3],
              "value_type": "float",  # Optional, defaults to inference from type of "bounds".
              "log_scale": False,  # Optional, defaults to False.
          },
          {
              "name": "pf_steps",
              "type": "choice",
              "values": [2,3,4,5],
              "is_ordered": False,
          },
          {
              "name": "x_data",
              "type": "range",
              "bounds": [1.0, 10.0],
          },
          {
              "name": "x_adv",
              "type": "range",
              "bounds": [0.0, 100.0],
          },
          {
              "name": "x_diff",
              "type": "range",
              "bounds": [0.0, 100.0],
          },
          {
              "name": "x_dt",
              "type": "range",
              "bounds": [0.0, 100.0],
          },
          {
              "name": "x_stab",
              "type": "range",
              "bounds": [0.0, 100.0],
          },
      ]

  def initialize(config, parameters):

      with open(config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
      config_dictionary = {**config, **parameters}


      print(jax.devices())
      print(jax.local_device_count())


      id = wandb.util.generate_id()


      wandb.init(project="turbulence_surrogates", entity="deepl4cfd", config=config_dictionary, id=id, resume='allow')

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
        model =  FNO2d(output_features = wandb.config.output_features, T=4)
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
      key, subkey =  jax.random.split(key,2)
      if wandb.config.model == 'CAE_LSTM':
        (_, carry), params  = model.init_with_output(subkey, field[0,:128,:128][...,:wandb.config.input_steps].numpy(), (None,None))
      else:
        params = model.init(subkey, field[0,:128,:128][...,:wandb.config.input_steps].numpy())
        carry = None
      del field

      alpha = wandb.config['lr']
      epoch_start = 0
      state = create_state(params, alpha, model, checkpoint=wandb.config.checkpoint, steps_per_epoch = len(dm.train_dataloader()), carry = carry)

      return state, dm

  def train_and_evaluate(parameterization):
    state, dm = initialize(args.config,parameterization)
    results = train(state, dm)
    return results

  if not args.checkpoint:
    ax_client.create_experiment(
      name=args.name,
      parameters=parameters,
      objectives={"validation_pde_loss": ObjectiveProperties(minimize=True),
                  "validation_relative_RMSE": ObjectiveProperties(minimize=True)},
      #parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
      #outcome_constraints=["validation_relative_RMSE <= 0.4"],  # Optional.
    )

  for i in range(args.n_trials):
    parameters, trial_index = ax_client.get_next_trial()
     # run the trials, but skip bad ones
    try:
      data = train_and_evaluate(parameters)
      ax_client.complete_trial(trial_index=trial_index, raw_data=data)
    except Exception:
      ax_client.abandon_trial(trial_index=trial_index)
      continue
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.save_to_json_file(filepath=args.save_path+'/'+args.name+'.json')  # For custom filepath, pass `filepath` argument.

if __name__ == "__main__":
  parser = ArgumentParser(add_help=False)
  parser = ArgumentParser(add_help=False)
  parser.add_argument('--config', type=str)
  parser.add_argument('--name', type=str)
  parser.add_argument('--save_path', type=str)
  parser.add_argument('--checkpoint', action='store_true')
  parser.add_argument('--n_trials', type=int, default=5)
  args = parser.parse_args()
  main(args)
