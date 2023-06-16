import wandb
import yaml
import os

os.environ['WANDB_API_KEY']='$37369cac70415978f6e8f2f17abb6feb280572a8'
os.environ['WANDB_MODE']="offline"
os.environ['WANDB_DIR']='/mnt/wandb'

with open("/mnt/DL4turb/config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

sweep_id = wandb.sweep(sweep = config, project='turbulence_surrogates')

wandb.agent(sweep_id)