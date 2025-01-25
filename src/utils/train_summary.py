import torch
from pathlib import Path
import os
import sys
#Insert current working directory to path variables => No relative path usage from Python 3.
sys.path.insert(0, os.getcwd())
from src.utils.misc import get_device_available
class Loss_tuple(object):
    def __init__(self):
        self.train = []
        self.val = []

def init_loss_dict(loss_name_list, history_loss_dict = None):
    loss_dict = {}
    for name in loss_name_list:
        loss_dict[name] = Loss_tuple()
    loss_dict['epochs'] = 0

    if history_loss_dict is not None:
        for k, v in history_loss_dict.items():
            loss_dict[k] = v

        for k, v in loss_dict.items():
            if k not in history_loss_dict:
                lt = Loss_tuple()
                lt.train = [0] * history_loss_dict['epochs']
                lt.val = [0] * history_loss_dict['epochs']
                loss_dict[k] = lt

    return loss_dict

def save_ckpt(model, optimizer, epoch, loss_dict, save_dir):
  #Save checkpoints every epoch
  if not Path(save_dir).exists():
      Path(save_dir).mkdir(parents=True, exist_ok=True) 
  ckpt_file = Path(save_dir).joinpath(f"epoch_{epoch}.tar")

  torch.save({
      'epoch': epoch,
      'loss_dict': loss_dict, #{loss_name: [train_loss_list, val_loss_list]}
      'model_state_dict': model,
      'optimizer_state_dict': optimizer,
  }, ckpt_file.absolute().as_posix())   

def load_ckpt(ckpt_path):
    #Ensure the loaded modules are inserted at the proper location on local device
    ckpt = torch.load(ckpt_path, map_location=get_device_available())

    # Retrieve the training parameters
    epoch = ckpt['epoch']
    loss_dict = ckpt['loss_dict']
    model_state_dict = ckpt['model_state_dict']
    optimizer_state_dict = ckpt['optimizer_state_dict']
    
    return epoch, loss_dict, model_state_dict, optimizer_state_dict

def write_summary(summary_writer, loss_dict, train_flag=True):
  curr_loss = loss_dict.copy()
  if (train_flag):
    for k, v in curr_loss.items():
      #Exclude k = epochs when writing to tensorboard
      if (k != 'epochs'):
        summary_writer.add_scalars(k, {'train': v.train[-1]}, len(v.train))
  else:
    for k, v in curr_loss.items():
      #Exclude k = epochs when writing to tensorboard
      if (k != 'epochs' ):
        summary_writer.add_scalars(k, {'val': v.val[-1]}, len(v.val))

def write_summary_comparison(summary_writer, loss_dict, train_flag=True):
  #This function is used for compare between original model and SCP model
  curr_loss = loss_dict.copy()
  if (train_flag):
    for k, v in curr_loss.items():
      #Exclude k = epochs when writing to tensorboard
      if (k != 'epochs'):
        summary_writer.add_scalars(k, {'train_original': v.train[-1]}, len(v.train))
  else:
    for k, v in curr_loss.items():
      #Exclude k = epochs when writing to tensorboard
      if (k != 'epochs' ):
        summary_writer.add_scalars(k, {'val_original': v.val[-1]}, len(v.val))