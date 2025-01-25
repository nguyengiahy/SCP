import torch

import os
import sys
#Insert current working directory to path variables => No relative path usage from Python 3.
sys.path.insert(0, os.getcwd())
from src.utils.train_summary import init_loss_dict, load_ckpt
from src.datasets.dataset import KTHDataset, get_dataloader
from src.utils.misc import get_device_available
from src.models.model import build_model
from src.training.train_function import single_iter
from src.utils.criterion import MSE

device = get_device_available()

N = 5
d_model = 768
resume_ckpt = "./checkpoint/checkpoint_6_heads_6_epochs.tar"
model = build_model(d_model, N)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
# Load the checkpoint
start_epoch, loss_dict, model_state_dict, optimizer_state_dict = load_ckpt(resume_ckpt)
# Load the model and the optimizer
model.to(device)
optimizer

root_dir = "./data"
test_split = 0.1
batch_size = 32
dataset = KTHDataset(root_dir, N)
_, _, test_loader = get_dataloader(dataset, batch_size, 1 - 2 * test_split, test_split, test_split)

loss_name = ['Total', 'MSE', 'SCL']
loss_dict = init_loss_dict(loss_name)

mse_loss = MSE()

for idx, batch in enumerate(test_loader, 0):
    temp_loss_dict = single_iter(model, optimizer, batch, device, mse_loss, train_flag=False)
    for k, v in temp_loss_dict.items():
        if k != 'epochs':
            loss_dict[k].val.append(temp_loss_dict[k])
    
for k, v in loss_dict.items():
    if (k != 'epochs'):
            loss_concat = torch.stack(v.val)
            loss_dict[k].val.append(torch.mean(loss_concat))

print(f"MSE loss of the model: {loss_dict['MSE'].val[0]}")
