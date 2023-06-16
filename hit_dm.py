import torch
import xarray
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

class mydataset(torch.utils.data.Dataset):
  def __init__(self,data, seq_size):
    self.data = data
    nx = len(self.data['x'])
    ny = len(self.data['y'])
    nz = len(self.data['z'])
    self.data = self.data
    self.seq_size = seq_size
    self.mean = [self.data['u'].mean().values, self.data['v'].mean().values, self.data['w'].mean().values, self.data['p'].mean().values]
    self.var = [self.data['u'].var().values, self.data['v'].var().values, self.data['w'].var().values, self.data['p'].var().values]
    self.total_k = (self.var[0] + self.var[1] + self.var[2])**0.5
    self.min = [self.data['u'].min().values, self.data['v'].min().values, self.data['w'].min().values, self.data['p'].min().values]
    self.max = [self.data['u'].max().values, self.data['v'].max().values, self.data['w'].max().values, self.data['p'].max().values]

  def __len__(self):
    return len(self.data['time']) // self.seq_size

  def __getitem__(self,idx):
    field = self.data.isel(time=slice(idx * self.seq_size, (idx+1) * self.seq_size ))
    return np.stack([self.norm_vel(field['u'].values,0),self.norm_vel(field['v'].values,1),
                    self.norm_vel(field['w'].values,2),self.norm_p(field['p'].values)], axis=-1)
  
  def norm_vel(self, x, i):
    x = (x - self.mean[i]) / self.total_k   # Normalize by total kinetic energy
    x = (x - self.min[i]) / (self.max[i]- self.min[i]) # minmax normalize
    return x
  
  def norm_p(self, x):
    x = (x - self.mean[-1]) / self.var[-1]**0.5 # Standard Normalization
    x = (x - self.min[-1]) / (self.max[-1]- self.min[-1]) # minmax normalize
    return x
      
class hit_dm(pl.LightningDataModule):
  
  def __init__(self,path,batch_size, seq_size):
    super().__init__()
    self.path = path
    self.seq_size = seq_size
    self.batch_size = batch_size
    
  def prepare_data(self):
    return None
  def setup(self,stage=None):
    data = xarray.open_mfdataset(self.path+'/*.nc', engine='netcdf4', combine='by_coords', parallel=True, decode_cf=False)
    data = xarray.decode_cf(data)
    N = len(data['time'])
    self.train_data, self.val_data = [mydataset(data.isel(time=slice(0,N*9//10)), self.seq_size),  mydataset(data.isel(time=slice(N*9//10,N)), self.seq_size)]

  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0)

  def test_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0)