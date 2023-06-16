import jax.numpy as jnp
from jax import debug, lax
import jax
import numpy as np

def lossl2(x, y):
  loss = jnp.mean((y-x)**2, axis=(0,1))
  return lax.pmean(loss, axis_name='v')

def rel_mse(x,y):
    loss = jnp.sum((y-x)**2, axis=(0,1,2)) / jnp.sum(y**2, axis=(0,1,2))
    return lax.pmean(loss**.5, axis_name='v')

def get_grid(shape):
        size_x, size_y = shape[0], shape[1]
        gridx, gridy = jnp.meshgrid(jnp.linspace(0, 1, size_x), jnp.linspace(0, 1, size_y))
        gridx = gridx.reshape(size_x, size_y, 1)
        gridy = gridy.reshape(size_x, size_y, 1)
        return jnp.concatenate((gridx, gridy), axis=-1)

def get_derivatives(w):
    nx = w.shape[0]
    ny = w.shape[1]
    nt = w.shape[-1]
    
    # Wavenumbers in y-direction
    k_max = nx//2
    kt_max = nt//2
    N = nx
    w_h = jnp.fft.fftn(w, axes=[0, 1])
    k_x = jnp.repeat(jnp.concatenate((jnp.arange(start=0, stop=k_max, step=1),
                     jnp.arange(start=-k_max, stop=0, step=1))).reshape(N, 1), repeats=N, axis=1).reshape(N,N,1)
    k_y = jnp.repeat(jnp.concatenate((jnp.arange(start=0, stop=k_max, step=1),
                     jnp.arange(start=-k_max, stop=0, step=1))).reshape(1, N), repeats=N, axis=0).reshape(N,N,1)
    k_t = jnp.concatenate((jnp.arange(start=0, stop=kt_max, step=1),
                     jnp.arange(start=-kt_max, stop=0, step=1))).reshape(1,1,-1,kt_max *2)
    # Negative Laplacian in Fourier space
    lap = -(k_x ** 2 + k_y ** 2)
    lap = lap.at[0, 0].set(1.0)
    f_h = w_h / -lap
    ux_h = 1j *  k_y * f_h
    uy_h = -1j *  k_x * f_h
    wx_h = 1j *  k_x * w_h
    wy_h = 1j *  k_y * w_h
    uxx_h = 1j * k_x * ux_h
    uyy_h = 1j * k_y * uy_h
    wlap_h = (0.01 * lap - 0.1) * w_h
    #ut_h = 1j * k_t * w_h

    ux = jnp.fft.irfftn(ux_h[:,:k_max+1], axes=[0, 1])
    uy = jnp.fft.irfftn(uy_h[:,:k_max+1], axes=[0, 1])
    uxx = jnp.fft.irfftn(uxx_h[:, :k_max + 1], axes=[0, 1])
    uyy = jnp.fft.irfftn(uyy_h[:, :k_max + 1], axes=[0, 1])
    wx = jnp.fft.irfftn(wx_h[:,:k_max+1], axes=[0,1])
    wy = jnp.fft.irfftn(wy_h[:,:k_max+1], axes=[0,1])
    wlap = jnp.fft.irfftn(wlap_h[:,:k_max+1], axes=[0,1])
    #ut = jnp.fft.irfftn(ut_h[..., :kt_max+1], axes=[0,1,-1])

    dt = 1.0
    ut = jnp.gradient(w,dt,axis=-1)

    return jnp.stack((ux, uy, uxx, uyy, wx, wy, wlap, ut))
    #diffusion_term = wlap 
    #advection_term = ux * wx + uy * wy
  
    #return jnp.stack((diffusion_term, advection_term, ut))


def pde_vor_loss(y,w):
  derivatives_w = get_derivatives(w)
  derivatives_y = get_derivatives(y)  
  return lax.pmean(jnp.mean((derivatives_w-derivatives_y)**2, axis=(0,1,2)), axis_name='v')

def pde_loss(x,y):
  x_h = jnp.fft.rfftn(x,axes=[0,1,-1])
  y_h = jnp.fft.rfftn(y,axes=[0,1,-1])
  n = (x_h-y_h).real**2 + (x_h-y_h).imag**2
  y_h = ((y_h).real**2 + (y_h).imag**2)
  D_ = lambda x: jnp.stack([jnp.gradient(x, axis=-1), jnp.gradient(x, axis=0), jnp.gradient(x, axis=1),
                 jnp.gradient(jnp.gradient(x, axis=0), axis=0), jnp.gradient(jnp.gradient(x, axis=1), axis=1)])
  D_x = D_(x)
  D_y = D_(y)
  return lax.pmean(jnp.sum(n)/jnp.sum(y_h) + jnp.mean((D_x-D_y)**2), axis_name='v')

def pi_loss(x):
  x = x[...,0,:]
  lx, ly, t = x.shape
  diff, adv, dt = jnp.split(get_derivatives(x), 3)
  X, Y = jnp.meshgrid(np.linspace(0, 2*np.pi, num=lx),np.linspace(0, 2*np.pi, num=ly))
  force = (jnp.cos(4 * Y))[..., np.newaxis]
  return lax.pmean(jnp.mean((dt[0] + adv[0] - diff[0] + force)**2, axis=(0,1)), axis_name='v')

def causality_loss(loss_fn, epsilon, w, *args):
  
  l_t = w * jax.vmap(loss_fn, axis_name='v')(*args)
  w = jnp.exp(-1 * epsilon * jnp.cumsum(l_t[0]))
  w = jax.lax.stop_gradient(w)
  loss = w * jax.vmap(loss_fn, axis_name='v')(*args)
  return loss[0].mean(), w
