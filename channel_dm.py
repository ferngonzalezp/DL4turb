import torch
import xarray
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

class mydataset(torch.utils.data.Dataset):
  def __init__(self,data):
    self.data = data
    nx = len(self.data['x'])
    ny = len(self.data['y'])
    self.min = [self.data['u'].min().values, self.data['v'].min().values, self.data['w'].min().values, self.data['p'].min().values]
    self.max = [self.data['u'].max().values, self.data['v'].max().values, self.data['w'].max().values, self.data['p'].max().values]
  def __len__(self):
    return len(self.data['batch'])

  def __getitem__(self,idx):
    field = self.data.isel(batch=[idx])
    return np.stack([self.norm(field['u'].values[0],0),self.norm(field['v'].values[0],1),
                    self.norm(field['w'].values[0],2),self.norm(field['p'].values[0],3)], axis=2)

  def norm(self, x, i):
    return (x - self.min[i]) / (self.max[i] - self.min[i])

class channel_dm(pl.LightningDataModule):
  
  def __init__(self,path,batch_size):
    super().__init__()
    self.path = path
    self.batch_size = batch_size
    
  def prepare_data(self):
    return None
  def setup(self,stage=None):
    data = xarray.open_mfdataset(self.path+'/*.nc', engine='h5netcdf', combine='by_coords', parallel=True, decode_cf=False)
    data = xarray.decode_cf(data)
    N = len(data['batch'])
    self.train_data, self.val_data = [mydataset(data.isel(batch=slice(0,N*9//10))),  mydataset(data.isel(batch=slice(N*9//10,N)))]

  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0)

  def test_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0)