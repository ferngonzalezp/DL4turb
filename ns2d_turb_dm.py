import torch
import xarray
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class mydataset(torch.utils.data.Dataset):
  def __init__(self,data):
    self.data = data
    nx = len(self.data['x'])
    ny = len(self.data['y'])
    self.data = self.data
    self.min = data['w'].min().values
    self.max = data['w'].max().values
  def __len__(self):
    return len(self.data['batch'])

  def __getitem__(self,idx):
    return (self.data.isel(batch=[idx])['w'].values[0] - self.min) / (self.max - self.min)
      
class ns2d_dm(pl.LightningDataModule):
  
  def __init__(self,path,batch_size, test_all=False):
    super().__init__()
    self.path = path
    self.batch_size = batch_size
    self.test_all = test_all
    
  def prepare_data(self):
    return None
  def setup(self,stage=None):
    data = xarray.open_mfdataset(self.path+'/*.nc', engine='netcdf4', concat_dim='batch', combine='nested', parallel=True)
    N = len(data['batch'])
    self.train_data, self.val_data = [mydataset(data.isel(batch=slice(0,N*9//10))),  mydataset(data.isel(batch=slice(N*9//10,N)))]
    if self.test_all:
      self.test_data = mydataset(data)
    else:
      self.test_data = self.val_data

  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0)

  def test_dataloader(self):
    return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=0)