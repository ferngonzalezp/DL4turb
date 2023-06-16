import jax
import jax.numpy as jnp
import numpy as np
from ns2d_turb_dm import ns2d_dm
import flax
from jax import lax, vmap, pmap
from flax.core import freeze, unfreeze
import functools
from fluid_anim import fluid_anim
from UFNET2 import UFNET as UFNET2
from FNO import FNO2d
from UNO import UNO
from test.get_model import get_model
from argparse import ArgumentParser
from loss import pde_vor_loss
from loss import rel_mse
from test.plot_fields import plot_fields
from test.compute_tke_spectrum import compute_tke_spectrum
from timeit import default_timer
import sys
import os
import random
import matplotlib.pyplot as plt
from compute_tke_spectrum import compute_tke_spectrum, plot_tkespec_1d

def main(args):

  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
  os.environ['XLA_FLAGS']='--xla_gpu_strict_conv_algorithm_picker={}'.format('false')
  dm = ns2d_dm(args.data_path, batch_size=args.batch_size, test_all=True)
  dm.setup()
  input_steps = args.input_steps
  pf_steps = 3

  if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
  
  key = jax.random.PRNGKey(10)
  shape = next(iter(dm.test_dataloader())).shape
  if args.modes:
    state =  get_model(args.model, key, shape[1:], 
                    args.state_path,
                    1, 1,
                    modes = args.modes)
  else:
    state =  get_model(args.model, key, shape[1:], 
                    args.data_path,
                    1, 1)
  
  param_count = sum(x.size for x in jax.tree_leaves(state.params))
  print('number of params:\n', param_count)

  def predict(params, input, T, teacher_forcing=None, ss = 0.5, push_forward=None, key_dropout = None, pred_delta=False):
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

  @functools.partial(pmap, static_broadcasted_argnums=(2,3,4), axis_name='p')
  def val_step(state, input, pred_delta=False, teacher_forcing=None, push_forward=None):
      T = input.shape[-1] - args.input_steps
      pred = vmap(predict, in_axes=(None, 0,None,None,None,None, None, None))(state.params, input, T, teacher_forcing, 1, push_forward, None, pred_delta)
      loss = vmap(rel_mse, axis_name='v')(pred[...,args.input_steps:], input[...,args.input_steps:])
      loss_pde = vmap(pde_vor_loss, axis_name='v')(pred[...,0,args.input_steps:], input[...,0,args.input_steps:])
      
      return lax.pmean(loss[0], axis_name='p'), pred, lax.pmean(loss_pde[0], axis_name='p')
  
  @functools.partial(pmap, static_broadcasted_argnums=(2,3,4,5), axis_name='p')
  def predict_long_T(state, input, T, teacher_forcing=None, push_forward=None, pred_delta=None):
    pred = vmap(predict, in_axes=(None,0,None,None,None,None, None, None))(state.params, input, T, teacher_forcing, 1, push_forward, None, pred_delta)
    return pred
  
  @jax.jit
  @functools.partial(vmap, axis_name='v')
  def autocorr(x):
      return jax.device_put(jax.lax.pmean(jax.scipy.signal.correlate(x,x,'full',)[-len(x):], axis_name='v'), jax.devices("cpu")[0])
  
  def test_loop():
    num_devices = jax.device_count()
    t1 = default_timer()
    val_loss = []
    pde_loss = []
    preds = []
    real = []
    pred_l = []
    for i, inputs in enumerate(dm.test_dataloader()):
            inputs = inputs[:,::args.ds,::args.ds].numpy()
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
              inputs = inputs.reshape(num_devices,bs//num_devices,size_x,size_y,args.output_features,-1)
              val_loss_step, _, pde_loss_step = val_step(state,inputs[:,:,nx-l:nx,ny-l:ny],args.pred_delta,True,None)
              pred_l.append(jax.device_put(predict_long_T(state, inputs, 999, None, None, args.pred_delta), jax.devices('cpu')[0]))
              val_loss.append(val_loss_step[0])
              pde_loss.append(pde_loss_step[0])
              #preds.append(jax.device_put(outputs[1], jax.devices('cpu')[0]))
              real.append(jax.device_put(inputs[:,:,nx-l:nx,ny-l:ny], jax.devices('cpu')[0]))
    t2 = default_timer()
    fig = []
    field_names = ['$\omega$']
    real = np.concatenate(real, axis=1)
    pred_l = np.concatenate(pred_l, axis=1)
    for i in range(args.output_features):
        fig.append(plot_fields(real[0,...,i,:],pred_l[0,...,i,:real.shape[-1]],field_names[i]))
    val_loss = np.mean(np.concatenate(val_loss, axis=0), axis=0)
    pde_loss = np.mean(np.concatenate(pde_loss, axis=0), axis=0)
    print('elapsed time: {} '.format(t2-t1))
    print('Val loss: {} PDE_loss: {}'.format(val_loss.mean(), pde_loss.mean()))
    return val_loss, pde_loss, fig, real, pred_l

  Val_loss, pde_loss, fig, real, pred_l = test_loop()
  #np.save(args.save_path+'/preds_'+args.model, preds)
  #np.save(args.save_path+'/preds_1000t_'+args.model, pred_l)
  fig[0].savefig(args.save_path+'/Preds_'+args.model+'.png')
  #fluid_anim(pred_l[0,0,:,:,0],args.save_path+'/pred_'+args.model)
  fluid_anim(pred_l[0,0,:,:,0],args.save_path+'/pred_1000t_'+args.model)
  fluid_anim(real[0,0,:,:,0],args.save_path+'/real_'+args.model)

  bs = 10
  T = real.shape[-1]
  result = []
  unnormalize = dm.unnormalize
  def normalize(x):
    return (x - np.mean(x, axis=-1)[...,np.newaxis])
    
  for i in range(real.shape[1]//bs):
    Ruu_real = autocorr(normalize(unnormalize(real[:,i*bs:bs*(i+1)])).reshape(-1,T))
    Ruu_pred = autocorr(normalize(unnormalize(pred_l[:,i*bs:bs*(i+1),...,:T])).reshape(-1,T))
    result.append([Ruu_real[0:1], Ruu_pred[0:1]])
  Ruu_real = np.mean(np.concatenate(result[:][0]), axis=0)
  Ruu_pred = np.mean(np.concatenate(result[:][1]), axis=0)
  Ruu_real = Ruu_real / Ruu_real.max()
  Ruu_pred = Ruu_pred / Ruu_pred.max()

  time = dm.test_data.data['time'].values
  tau = np.zeros(len(time)*2-1)
  tau[0:len(time)] = -time[-1] + time
  tau[len(time):] = time[1:] - time[0]
  print(time[1]-time[0])
  fig = plt.figure(figsize=(6, 3.6), dpi=200, constrained_layout=True)
  plt.plot(tau[-len(time):], Ruu_real, label='Simulation, T=50s')
  plt.plot(tau[-len(time):], Ruu_pred, 'o', markersize='2', markevery=5, markerfacecolor='w', label=args.model + ' T=50s')
  plt.xlabel('$ \\tau $ [s]')
  plt.ylabel('$R_{uu}$ [$m^2/s^2$]')
  plt.grid()
  plt.legend()
  plt.rc("font", size=10, family='serif')
  plt.show()
  fig.savefig(args.save_path + '/R_'+args.model+'_Re100.png')
  
  get_psd = lambda x: jax.scipy.signal.welch(x, window=('hann'), nperseg=T)

  psd_real = []
  psd_pred_l = []
  psd_pred = []
  for i in range(real.shape[1]//10):
    psd_real.append(jax.device_put(vmap(get_psd)(real[:,i*10:(i+1)*10].reshape(-1,T)), jax.devices('cpu')[0]))
    psd_pred_l.append(jax.device_put(vmap(get_psd)(pred_l[:,i*10:(i+1)*10].reshape(-1,1000)), jax.devices('cpu')[0]))
    psd_pred.append(jax.device_put(vmap(get_psd)(pred_l[:,i*10:(i+1)*10,...,:T].reshape(-1,T)), jax.devices('cpu')[0]))
  f, psd_real = (psd_real[:][0][0][0], np.concatenate(psd_real[:][1]))
  psd_pred_l = np.concatenate(psd_pred_l[:][1])
  psd_pred = np.concatenate(psd_pred[:][1])

  fig = plt.figure(figsize=(6, 3.8), dpi=200, constrained_layout=True)
  plt.semilogy(f, np.mean(psd_real,axis=0) , label='Simulation, T=50s')
  plt.semilogy(f, np.mean(psd_pred,axis=0) , label=args.model + ' T=50s')
  plt.semilogy(f, np.mean(psd_pred_l,axis=0), label=args.model + ' T=250s')
  plt.xlabel('frequency [Hz]')
  plt.ylabel('PSD [$(m^2/s^2)$/Hz]')
  plt.grid()
  plt.legend()
  plt.rc("font", size=10, family='serif')
  plt.show()
  fig.savefig(args.save_path+'/PSD_'+args.model+'_Re100.png')
  
  T = np.arange(10, real.shape[-1], (real.shape[-1] - 10)//3 - 1)
  E_real = [0.0] * len(T)
  E_pred = [0.0] * len(T)
  E_pred_l = [0.0] * len(T)

  for i in range(len(T)):
    for j, batch in enumerate(dm.test_dataloader()):  
      real = batch.numpy()
      knyquist, k, E1 = compute_tke_spectrum(real[:,::2,::2,i],2*np.pi,2*np.pi,True)
      knyquist, k, E2 = compute_tke_spectrum(pred_l[0,j*args.batch_size:(j+1)*args.batch_size,...,0,i],2*np.pi,2*np.pi,True)
      E_real[i] += E1[0] / pred_l.shape[0]
      E_pred[i] += E2[0] / pred_l.shape[0]

    E = [{'spec': E_real[i], 'color': '-k', 'label':'Sim'},
      {'spec': E_pred[i], 'color': '-bo', 'label':args.model}]
    fig = plot_tkespec_1d(knyquist[0],k[0],E, 'TKE Spectrum T = {}$\delta t$'.format(T[i]))
    fig.savefig(args.save_path + '/tke_{}_T{}.png'.format(args.model, T[i]))

if __name__ == "__main__":

  parser = ArgumentParser(add_help=False)
  parser = ArgumentParser(add_help=False)
  parser.add_argument('--batch_size', type=int, default=4)
  parser.add_argument('--output_features', type=int, default=1)
  parser.add_argument('--input_steps', type=int, default=1)
  parser.add_argument('--modes', type=int, default=None)
  parser.add_argument('--ds', type=int, default=2)
  parser.add_argument('--data_path', type=str)
  parser.add_argument('--state_path', type=str)
  parser.add_argument('--save_path', type=str, default = './')
  parser.add_argument('--model', type=str)
  parser.add_argument('--pred_delta', action='store_true')
  args = parser.parse_args()

  main(args)