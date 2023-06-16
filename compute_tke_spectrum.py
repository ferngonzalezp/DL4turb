import numpy as np
from jax.numpy.fft import fftn
from jax.numpy import sqrt, zeros, conj, arange, ones
from numpy import pi
from scipy.signal import convolve
import functools
import jax
from jax import vmap
from jax import lax
import sys
import os
import random
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from ns2d_turb_dm import ns2d_dm


# ------------------------------------------------------------------------------

def movingaverage(interval, window_size):
    window = ones(int(window_size)) / float(window_size)
    return convolve(interval, window)


# ------------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(1,2,3))
@functools.partial(vmap, in_axes=(0,None,None,None), axis_name='v')
def compute_tke_spectrum(u, lx, ly, one_dimensional = True):
    """
    Given a velocity field u this function computes the kinetic energy
    spectrum of that velocity field in spectral space. This procedure consists of the
    following steps:
    1. Compute the spectral representation of u using a fast Fourier transform.
    This returns uf (the f stands for Fourier)
    2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf)* conjugate(uf)
    3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy
    Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
    the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
    E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

    Parameters:
    -----------
    u: 2D array
      The x-velocity component.
    lx: float
      The domain size in the x-direction.
    ly: float
      The domain size in the y-direction.
    smooth: boolean
      A boolean to smooth the computed spectrum for nice visualization.
    """
    nx = len(u[:, 0])
    ny = len(u[0, :])

    nt = nx * ny
    n = max(nx, ny)  # int(np.round(np.power(nt,1.0/3.0)))

    uh = fftn(u) / nt

    # tkeh = zeros((nx, ny, nz))
    tkeh = 0.5 * (uh * conj(uh)).real

    length = max(lx, ly)

    knorm = 2.0 * pi / length

    kxmax = nx / 2
    kymax = ny / 2

    wave_numbers = knorm * arange(0, n)
    tke_spectrum = zeros(len(wave_numbers))
    if one_dimensional == True:
      for kx in range(nx):
          rkx = kx
          if kx > kxmax:
              rkx = rkx - nx
          for ky in range(ny):
              rky = ky
              if ky > kymax:
                  rky = rky - ny
              rk = ((rkx * rkx + rky * rky))**0.5
              k = int(np.round(rk))
              tke_spectrum = tke_spectrum.at[k].set(tke_spectrum[k] + tkeh[kx, ky])

      tke_spectrum = tke_spectrum / knorm
    else:
      tke_spectrum = tkeh

    knyquist = knorm * min(nx, ny) / 2

    return knyquist, wave_numbers, lax.psum(tke_spectrum, axis_name='v')

# ---------------------------------------------------------------------------------------    

def plot_tkespec_1d(knyquist, wavenumbers, tkespec, title):
    plt.rc("font", size=10, family='serif')

    fig = plt.figure(figsize=(6, 4.8), dpi=200, constrained_layout=True)
    l = []
    for i in range(len(tkespec)):
      l1, = plt.loglog(wavenumbers, tkespec[i]['spec'], tkespec[i]['color'], label=tkespec[i]['label'], markersize=3, markerfacecolor='w', markevery=5)
      l.append(l1)
    plt.axis([0.9, 100, 1e-18, 100])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axvline(x=knyquist, linestyle='--', color='black')
    plt.xlabel('$\kappa$ (1/m)')
    plt.ylabel('$E(\kappa)$ (m$^3$/s$^2$)')
    plt.grid()
    #plt.gcf().tight_layout()
    plt.title(title + str(len(wavenumbers)) + 'x' + str(len(wavenumbers)))
    plt.legend(handles=l, loc=3)
    return fig


def main(args):

  dm = ns2d_dm(args.data_path, batch_size=args.batch_size, test_all=False)
  dm.setup()
  preds = np.load(args.save_path+'/preds_'+args.model+'.npy')
  print(preds.shape)
  #pred_l = np.load(args.save_path+'/preds_1000t_'+args.model+'.npy')
  T = np.arange(10, preds.shape[-1], (preds.shape[-1] - 10)//3 - 1)
  E_real = [0.0] * len(T)
  E_pred = [0.0] * len(T)
  E_pred_l = [0.0] * len(T)

  for i in range(len(T)):
    for j, batch in enumerate(dm.test_dataloader()):  
      real = batch.numpy()
      knyquist, k, E1 = compute_tke_spectrum(real[:,::2,::2,i],2*np.pi,2*np.pi,True)
      knyquist, k, E2 = compute_tke_spectrum(preds[0,j*args.batch_size:(j+1)*args.batch_size,...,0,i],2*np.pi,2*np.pi,True)
      E_real[i] += E1[0] / preds.shape[0]
      E_pred[i] += E2[0] / preds.shape[0]

    E = [{'spec': E_real[i], 'color': '-k', 'label':'Sim'},
      {'spec': E_pred[i], 'color': '-bo', 'label':args.model}]
    fig = plot_tkespec_1d(knyquist[0],k[0],E, 'TKE Spectrum T = {}$\delta t$'.format(T[i]))
    fig.savefig(args.model + '/tke_{}.png'.format(args.model))

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