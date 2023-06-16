import random
import numpy as np
from jax import grad, jit, vmap, pmap
from jax import lax, numpy as jnp
import random
import time
import jax
from typing import Any, Callable, Sequence, Optional
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import optax
from flax.training.train_state import  TrainState
import functools
from timeit import default_timer
from flax.training import train_state
from flax.training import checkpoints
import os
import dataclasses
from loss import lossl2, pde_vor_loss,  pde_loss, get_derivatives, causality_loss, pi_loss
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def save_checkpoint(state, step_or_metric, prefix, save_path):
    """
    Saves a checkpoint from the given state.
    Args:
        state (train_state.TrainState): Training state.
        step_or_metric (int of float): Current training step or metric to identify the checkpoint.
        path (str): Path to the checkpoint directory.
    """
    if jax.process_index() == 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(save_path, state, step_or_metric, keep=2, prefix=prefix, overwrite=True)

def restore_checkpoint(state, path):
    """
    Restores checkpoint with best validation score.
    Args:
        state (train_state.TrainState): Training state.
        path (str): Path to checkpoint.
    Returns:
        (train_state.TrainState): Training state from checkpoint.
    """
    return checkpoints.restore_checkpoint(path, state)

class TrainState(train_state.TrainState):
    """
    Simple train state for the common case with a single Optax optimizer.
    Attributes:
        epoch (int): Current epoch.
    """
    epoch: int
    carry: any
    w: any
    epsilon: any
    dropout_key : jax.random.KeyArray

def create_state(params, alpha, model, checkpoint=None, prefix='ckpt_epoch', steps_per_epoch = 100, 
                carry=None, time_window = 10, epsilon= [1,1], dropout_key = None):
        #schedule_fn = optax.exponential_decay(alpha, 50*steps_per_epoch, 0.5, transition_begin=50*steps_per_epoch, staircase=False, end_value=1e-6)
        schedule_fn = optax.warmup_cosine_decay_schedule(alpha, 100*alpha, 10, 50*steps_per_epoch)
        state = TrainState.create(
            apply_fn=model.apply,
            params = params['params'],
            tx = optax.chain(
                        optax.clip(1.0),
                        optax.adam(learning_rate=alpha),
                      ),
            #tx = optax.adam(learning_rate=schedule_fn),
            epoch=0,
            carry = carry,
            w = jnp.ones((2, time_window-1)),
            epsilon = jnp.stack(epsilon),
            dropout_key = dropout_key)
        if checkpoint:
          ckpt_path = checkpoints.latest_checkpoint(checkpoint, prefix=prefix)
          state = restore_checkpoint(state, ckpt_path)
          epoch_start = state.epoch + 1
          state = dataclasses.replace(state, **{'epoch': epoch_start})
        return flax.jax_utils.replicate(state)

def train(state, dm):
  
  input_steps = wandb.config['input_steps']
  pf_steps = wandb.config['pf_steps']
  teacher_forcing = wandb.config['teacher_forcing']
  push_forward = wandb.config['push_forward']
  unroll_steps = wandb.config['unroll_steps']
  unroll_loss = wandb.config['unroll_loss']
  x_diff = wandb.config['x_diff']
  x_adv = wandb.config['x_adv']
  x_dt = wandb.config['x_dt']
  x_data = wandb.config['x_data']
  x_stab = wandb.config['x_stab']
  x_eq = wandb.config['x_eq']
  dataset = wandb.config['dataset']
  output_features = wandb.config['output_features']
  ds = wandb.config['ds']
  save_path = wandb.config['save_path']
  max_epochs = wandb.config['max_epochs']
  time_window = wandb.config['time_window']
  epsilon_data = wandb.config['epsilon_data']
  epsilon_eq = wandb.config['epsilon_eq']
  causality_train = wandb.config['causality_train']
  pred_delta = wandb.config['pred_delta']
  denoise = wandb.config['denoise']
  

  def predict(params, input, T, teacher_forcing=None, ss = 0.5, push_forward=None, key_dropout = None):
    y = jnp.array(input)
    input = jnp.array(input[...,:input_steps])
    pred = input
    def prediction(input, x):
      if pred_delta:
          delta = state.apply_fn({'params': params}, input)[...,np.newaxis]
          pred = delta + input[...,-1:]
      else:
          pred = state.apply_fn({'params': params}, input)[...,np.newaxis]
      input = jnp.concatenate(((input[...,1:],pred)), axis=-1)
      return input, pred
    
    scan_pred = lambda input, length: lax.scan(prediction, input, None, length=length)

    if push_forward:
      xs = []
      for i in range(pf_steps,T):
        xs.append(y[...,max(0,i-pf_steps):][...,:input_steps])
      xs = jnp.stack(xs)
      pred, _ = vmap(scan_pred, in_axes=(0,None))(xs, pf_steps)
      pred, _ = vmap(scan_pred, in_axes=(0,None))(lax.stop_gradient(pred), 1)
      pred = jnp.concatenate((y[...,:input_steps+pf_steps],np.swapaxes(pred,0,-1)[0]), axis=-1)     
    else:
      if teacher_forcing:
        xs = [input]
        for i in range(T-input_steps):
          idx = np.array(random.sample(range(input_steps), int(ss*input_steps)))
          xs.append(y[...,i+1+idx])
        input = jnp.stack(xs)
        pred, _ = vmap(scan_pred, in_axes=(0,None))(input, 1)
        pred = jnp.concatenate((y[...,:input_steps],np.swapaxes(pred,0,-1)[0]), axis=-1)
      else:
        _, pred = lax.scan(prediction, input, None, T)
        pred = jnp.concatenate((y[...,:input_steps],np.swapaxes(pred,0,-1)[0]), axis=-1)
    return pred

  def predict_lstm(params, input, T, teacher_forcing=None, ss = 0.5, push_forward=None, key_dropout = None):
    y = jnp.array(input)
    input = jnp.array(input[...,:input_steps])
    pred = input
    carry = (state.carry[0][0],state.carry[1][0])
    def prediction(input, x):
      pred, carry = state.apply_fn({'params': params},input[0], input[1])
      input = jnp.concatenate(((input[0][...,1:],pred[...,np.newaxis])), axis=-1)
      return (input, carry), pred[...,np.newaxis]
    
    scan_pred = lambda input, carry, length: lax.scan(prediction, (input, carry), None, length=length)

    if push_forward:
      xs = []
      for i in range(pf_steps,T):
        xs.append(y[...,max(0,i-pf_steps):][...,:input_steps])
      xs = jnp.stack(xs)
      (pred, carry), _ = vmap(scan_pred, in_axes=(0,None,None))(xs, carry, pf_steps)
      (pred, carry), _ = vmap(scan_pred, in_axes=(0,((0,0)),None))(lax.stop_gradient(pred),lax.stop_gradient(carry), 1)
      pred = jnp.concatenate((y[...,:input_steps+pf_steps],np.swapaxes(pred,0,-1)[0]), axis=-1)     
    else:
      if teacher_forcing:
        xs = [input]
        for i in range(T-input_steps):
          idx = np.array(random.sample(range(input_steps), int(ss*input_steps)))
          xs.append(y[...,i+1+idx])
        xs = jnp.stack(xs)
        (pred, carry), _ = vmap(scan_pred, in_axes=(0,None,None))(xs,carry,1)
        pred = jnp.concatenate((y[...,:input_steps],np.swapaxes(pred,0,-1)[0]), axis=-1)
      else:
        _, pred = lax.scan(prediction, (input, carry), None, T)
        pred = jnp.concatenate((y[...,:input_steps],np.swapaxes(pred,0,-1)[0]), axis=-1)
    return pred
  
  def predict_vit(params, input, T, teacher_forcing=None, ss = 0.5, push_forward=None, key_dropout = None):
    y = jnp.array(input)
    input = jnp.array(input[...,:input_steps])
    pred = input
    if key_dropout is not None:
      def prediction(input, x):
        if pred_delta:
          delta = state.apply_fn({'params': params}, input, training = True, rngs={'dropout': key_dropout})[...,np.newaxis]
          pred = delta + input[...,-1:]
        else:
          pred = state.apply_fn({'params': params}, input, training = True, rngs={'dropout': key_dropout})[...,np.newaxis]
        input = jnp.concatenate(((input[...,1:],pred)), axis=-1)
        return input, pred
    else:
      def prediction(input, x):
        if pred_delta:
          delta = state.apply_fn({'params': params}, input, training = False)[...,np.newaxis]
          pred = delta + input[...,-1:]
        else:
          pred = state.apply_fn({'params': params}, input, training = False)[...,np.newaxis]
        input = jnp.concatenate(((input[...,1:],pred)), axis=-1)
        return input, pred
    
    scan_pred = lambda input, length: lax.scan(prediction, input, None, length=length)

    if push_forward:
      xs = []
      for i in range(pf_steps,T):
        xs.append(y[...,max(0,i-pf_steps):][...,:input_steps])
      xs = jnp.stack(xs)
      pred, _ = vmap(scan_pred, in_axes=(0,None))(xs, pf_steps)
      pred, _ = vmap(scan_pred, in_axes=(0,None))(lax.stop_gradient(pred), 1)
      pred = jnp.concatenate((y[...,:input_steps+pf_steps],np.swapaxes(pred,0,-1)[0]), axis=-1)     
    else:
      if teacher_forcing:
        xs = [input]
        for i in range(T-input_steps):
          idx = np.array(random.sample(range(input_steps), int(ss*input_steps)))
          xs.append(y[...,i+1+idx])
        input = jnp.stack(xs)
        pred, _ = vmap(scan_pred, in_axes=(0,None))(input, 1)
        pred = jnp.concatenate((y[...,:input_steps],np.swapaxes(pred,0,-1)[0]), axis=-1)
      else:
        _, pred = lax.scan(prediction, input, None, T)
        pred = jnp.concatenate((y[...,:input_steps],np.swapaxes(pred,0,-1)[0]), axis=-1)
    return pred
  
  if wandb.config.model == 'CAE_LSTM':
    predict = predict_lstm
  
  if wandb.config.model == 'UVIT':
    predict = predict_vit

  def mapped_loss(params, input, dataset, key):

        y = input[...,input_steps:]
        w = vmap(predict, in_axes=(None,0, None, None, None, None, None))(params, input, y.shape[-1], teacher_forcing, 1, False, key)          
        w_pf = 0.0
        loss_stability = np.array((0.0,))
        if push_forward:
          w_pf = vmap(predict, in_axes=(None,0,None, None, None, None, None))(params, input, y.shape[-1], teacher_forcing, 1, push_forward, key)
          if pred_delta:
            pred = w_pf[...,pf_steps:][...,1:] - w_pf[...,pf_steps:][...,:-1]
            target = input[...,pf_steps:][...,1:] - input[...,pf_steps:][...,:-1]
            loss_stability = vmap(lossl2, axis_name='v')(pred, target)
          else:
            loss_stability = vmap(lossl2, axis_name='v')(w_pf[...,pf_steps+input_steps:], y[...,pf_steps:])
        elif unroll_loss:
           w_ur = vmap(predict, in_axes=(None,0, None, None, None, None, None))(params, input, unroll_steps, False, 1, False, key)
           loss_stability = vmap(lossl2, axis_name='v')(w_ur[...,input_steps:], y[...,:unroll_steps])
        elif denoise:
          noise = jax.random.multivariate_normal(key, np.array([0,0]), np.array([[0.01,0.01],[0.01,0.01]]), shape=input.shape)
          #scale =  np.random.Generator.random(inputs.shape[0])
          w_pf = vmap(predict, in_axes=(None,0,None, None, None, None, None))(params, input + noise[...,0], y.shape[-1], teacher_forcing, 1, None, key)
          if pred_delta:
            pred = w_pf[...,1:] - w_pf[...,:-1]
            target = input[...,1:] - input[...,:-1]
            loss_stability = vmap(lossl2, axis_name='v')(pred, target)
          else:
            loss_stability = vmap(lossl2, axis_name='v')(w_pf[...,input_steps:], y)
        if pred_delta:
          pred = w[...,1:] - w[...,:-1]
          target = input[...,1:] - input[...,:-1]
          loss_val = vmap(lossl2, axis_name='v')(pred, target)
        else:
          loss_val = vmap(lossl2, axis_name='v')(w[...,input_steps:], y)
        if '2dturb' in dataset:
          loss_res = vmap(pde_vor_loss, axis_name='v')(w[...,0,:], input[...,0,:])
        else:
          loss_res = vmap(pde_loss, axis_name='v')(w,input)
        output = x_data * loss_val[0].mean()  + x_eq * loss_res[0].mean() + x_stab * loss_stability[0].mean()
        return output, (loss_val[0].mean(), loss_res[0].mean(), loss_stability[0].mean())
  
  def total_causality_loss(params, input, w, epsilon, key):
      y = input[...,input_steps:]
      w1, w2 = jnp.split(w,2)
      epsilon_data, epsilon_eq = jnp.split(epsilon,2)          
      pred = vmap(predict, in_axes=(None,0, None, None, None, None, None))(params, input, y.shape[-1], teacher_forcing, 1, False, key)
      loss_data, w1 = causality_loss(lossl2, epsilon_data[0], w1[0], pred[...,input_steps:], y)
      loss_res, w2 = causality_loss(pde_vor_loss, epsilon_eq[0], w2[0], pred[...,0,input_steps:], y[...,0,:])
      w = jnp.stack([w1, w2])
      return loss_data + loss_res, ((loss_data, loss_res) , w)

  if causality_train:
    loss_grad_fn = jax.value_and_grad(total_causality_loss, argnums=0, has_aux=True)

    @functools.partial(jax.pmap, axis_name='p')
    def train_step(state, inputs, key):
              dropout_train_key = jax.random.fold_in(key=key, data=state.step)
              output, grads = loss_grad_fn(state.params, inputs, state.w, state.epsilon, dropout_train_key)
              grads =  lax.pmean(grads, axis_name='p')
              loss_step = lax.pmean(output[1][0], axis_name='p')
              state = state.apply_gradients(grads=grads)
              w = lax.pmean(output[1][1], axis_name='p')
              return state, loss_step, w

  else:
    loss_grad_fn = jax.value_and_grad(mapped_loss,argnums=0, has_aux=True)

    @functools.partial(jax.pmap, axis_name='p')
    def train_step(state, inputs, key):
              dropout_train_key = jax.random.fold_in(key=key, data=state.step)
              output, grads = loss_grad_fn(state.params, inputs, dataset, dropout_train_key)
              grads =  lax.pmean(grads, axis_name='p')
              loss_step = lax.pmean(output[1], axis_name='p')
              state = state.apply_gradients(grads=grads)
              return state, loss_step
  min = 100
  num_devices = jax.device_count()

  def rel_mse(x,y):
    loss = jnp.sum((y-x)**2, axis=(0,1,2)) / jnp.sum(y**2, axis=(0,1,2))
    return lax.pmean(loss**0.5, axis_name='v')

  @functools.partial(pmap, static_broadcasted_argnums=(2,3), axis_name='p')
  def val_step(state, input, teacher_forcing=None, push_forward=None):
    T = input.shape[-1] - input_steps
    pred = vmap(predict, in_axes=(None,0,None,None,None,None, None))(state.params, input, T, teacher_forcing, 1, push_forward, None)
    loss = vmap(rel_mse, axis_name='v')(pred[...,input_steps:], input[...,input_steps:])
    if '2dturb' in wandb.config.dataset:
      loss_pde = vmap(pde_vor_loss, axis_name='v')(pred[...,0,input_steps:], input[...,0,input_steps:])
    
    return lax.pmean(loss[0], axis_name='p'), pred, lax.pmean(loss_pde[0], axis_name='p')

  def plot_fields(field,pred):
    plt.rc("font", size=16, family='serif')
    N = field.shape[-1]
    idx = np.random.randint(field.shape[0])
    time_steps = np.arange(10,N+1,(N-10)//3)-1
    fig, axs = plt.subplots(2,4,figsize=(20,10))
    fig.suptitle('Ground truth vs. Prediction',fontsize=24)
    ax = axs[0,0]
    cm = ax.imshow(field[idx,...,time_steps[0]], cmap=sns.cm.icefire)
    ax.set_title('t = ' + str(time_steps[0]) + 's')
    ax.set_ylabel('Ground truth')
    ax = axs[0,1]
    cm = ax.imshow(field[idx,...,time_steps[1]], cmap=sns.cm.icefire)
    ax.set_title('t = ' + str(time_steps[1]) + 's')
    ax = axs[0,2]
    cm = ax.imshow(field[idx,...,time_steps[2]], cmap=sns.cm.icefire)
    ax.set_title('t = ' + str(time_steps[2]) + 's')
    ax = axs[0,3]
    cm = ax.imshow(field[idx,...,time_steps[3]], cmap=sns.cm.icefire)
    ax.set_title('t = ' + str(time_steps[3]) + 's')
    ax = axs[1,0]
    cm = ax.imshow(pred[idx,...,time_steps[0]], cmap=sns.cm.icefire)
    ax.set_ylabel('Prediction')
    ax = axs[1,1]
    cm = ax.imshow(pred[idx,...,time_steps[1]], cmap=sns.cm.icefire)
    ax = axs[1,2]
    cm = ax.imshow(pred[idx,...,time_steps[2]], cmap=sns.cm.icefire)
    ax = axs[1,3]
    cm = ax.imshow(pred[idx,...,time_steps[3]], cmap=sns.cm.icefire)
    fig.colorbar(cm, ax = axs)
    plt.close()
    return fig

  def evaluate():
    num_devices = jax.device_count()
    t1 = default_timer()
    val_loss = []
    val_loss_pde = []
    preds = []
    for i, inputs in enumerate(dm.val_dataloader()):
            inputs = inputs[:,::ds,::ds].numpy()
            bs = inputs.shape[0]
            size_x = inputs.shape[1]
            size_y = inputs.shape[2]
            r = np.max([size_x,size_y]) // np.min([size_x,size_y])
            if size_x != size_y:
              l = int(np.min([size_x,size_y]))
            else:
              l = size_x
            for n in range(r):
              nx = int(np.min([size_x, l*(n+1)]))
              ny = int(np.min([size_y, l*(n+1)]))
              inputs = inputs.reshape(num_devices,bs//num_devices,size_x,size_y,output_features,-1)
              outputs = val_step(state,inputs[:,:,nx-l:nx,ny-l:ny],None,None)
              val_loss.append(outputs[0][0])
              preds.append(outputs[1])
              val_loss_pde.append(outputs[2][0])
    t2 = default_timer()
    fig = plot_fields(inputs[0,:,nx-l:nx,ny-l:ny,0],outputs[1][0,:,:,:,0])
    val_loss = np.mean(np.stack(val_loss), axis=0)
    val_loss_pde = np.mean(np.stack(val_loss_pde), axis=0)
    print('elapsed time: {} '.format(t2-t1))
    print(val_loss.mean())
    print(val_loss_pde.mean())
    preds = np.concatenate(preds, axis=1)
    return val_loss, preds, fig, val_loss_pde
  wma = []
  for i in range(state.epoch[0], max_epochs):
        t1 = default_timer()
        epoch_loss = 0.0
        epoch_res = 0.0
        epoch_d = 0.0
        epoch_stab = 0.0
        nb = len(dm.train_dataloader())
        for j, inputs in enumerate(dm.train_dataloader()):
          T = inputs.shape[-1]
          inputs = inputs[:,::ds,::ds].numpy()
          bs = inputs.shape[0]
          size_x = inputs.shape[1]
          size_y = inputs.shape[2]
          r = np.max([size_x,size_y]) // np.min([size_x,size_y])
          if size_x != size_y:
            l = int(np.min([size_x,size_y]))
          else:
            l = size_x
          inputs = inputs.reshape(num_devices,bs//num_devices,size_x,size_y,output_features,-1)
          time_window = time_window
          for t in range(T//time_window):
            for n in range(r):
              nx = int(np.min([size_x, l*(n+1)]))
              ny = int(np.min([size_y, l*(n+1)]))
              if causality_train:
                state, output, w = train_step(state, inputs[:,:,nx-l:nx,ny-l:ny,:,time_window*t:time_window*(t+1)], state.dropout_key)
                if len(wma) == 10:
                  wma = wma[1:]
                wma.append(w)
                state = dataclasses.replace(state, **{'w': jnp.mean(jnp.stack(wma), axis=0)})
              else:
                state, output = train_step(state, inputs[:,:,nx-l:nx,ny-l:ny,:,time_window*t:time_window*(t+1)], state.dropout_key)
              loss_val = output[0]
              residual = output[1]
              loss_stability = output[2]
              #loss_stability = [0.0]
              epoch_loss += loss_val/(nb*r*T/time_window)
              epoch_res += residual/(nb*r*T/time_window)
              epoch_stab += loss_stability/(nb*r*T/time_window)
              print('epoch {}:'.format(state.epoch[0]),'loss step {}: '.format(state.step[0]), loss_val[0], 
                      'pde_loss step {}: '.format(state.step[0]), residual[0], 
                      'loss_stab step {}: '.format(state.step[0]), loss_stability[0], 
                      #'weights min data step {}: '.format(state.step[0]), state.w[0][0].min(),
                      #'weights min res step {}: '.format(state.step[0]), state.w[0][1].min(),
                      )
              
              if (state.w[0][0].min() > 0.99) and (state.epsilon[0][0] < 1000):
                state = dataclasses.replace(state, **{'epsilon': flax.jax_utils.replicate(state.epsilon[0].at[0].set(state.epsilon[0][0]*10))})
                print('New epsilon_data: {}'.format(state.epsilon[0][0]))
              if state.w[0][1].min() > 0.99 and (state.epsilon[0][1] < 1000):
                state = dataclasses.replace(state, **{'epsilon': flax.jax_utils.replicate(state.epsilon[0].at[1].set(state.epsilon[0][1]*10))})
                print('New epsilon_eq: {}'.format(state.epsilon[0][1]))  
        t2 = default_timer()

        
        print('epoch {}: '.format(state.epoch[0]),'elapsed time: {} '.format(t2-t1),'loss epoch: ', epoch_loss[0],
                'pde_loss epoch: ', epoch_res[0], 'stability_loss_epoch: ', epoch_stab[0])
        wandb.log({'training_loss': epoch_loss[0],
                    'pde_loss': epoch_res[0],
                    'stability_loss': epoch_stab[0]})

        
        save_checkpoint(state, i, prefix='ckpt_epoch_', save_path = wandb.run.dir)
        if epoch_loss[0] < min:
          save_checkpoint(state, epoch_loss[0], prefix='ckpt_min_', save_path = wandb.run.dir)
          min = epoch_loss[0]
          #wandb.save(os.path.join(wandb.run.dir, "ckpt_min.ckpt"), base_path=wandb.run.dir)
        #wandb.save(os.path.join(wandb.run.dir, "ckpt_epoch.ckpt"), base_path=wandb.run.dir)

        print('Evaluating...')
        val_loss, _, fig, val_loss_pde = evaluate()
        image = wandb.Image(fig)
        plt.plot(val_loss)
        plt.ylabel('RMSE')
        plt.xlabel('time (s)')
        plt.title('RMSE(t)')
        print('epoch {}: '.format(state.epoch[0]),'elapsed time: {} '.format(t2-t1),'Val loss: ', val_loss.mean(),
         'val_loss_pde: ', val_loss_pde.mean())
        wandb.log({'validation relative RMSE': val_loss.mean(),
                    'validation_pde_loss': val_loss_pde.mean(),
                    'Samples': image, 'RMSE vs time': wandb.Image(plt)})
        plt.close()
        state = dataclasses.replace(state, **{'epoch': flax.jax_utils.replicate(i+1)})

  return {'validation_relative_RMSE': (val_loss.mean(), 0.0),
          'validation_pde_loss': (val_loss_pde.mean(), 0.0)}

        

