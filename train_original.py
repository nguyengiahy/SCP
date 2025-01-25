import os
import sys
#Insert current working directory to path variables => No relative path usage from Python 3.
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from pathlib import Path
import logging
from datetime import datetime

from src.utils.misc import set_seed, get_device_available
from src.utils.train_summary import init_loss_dict, load_ckpt, write_summary, write_summary_comparison, save_ckpt
from src.datasets.dataset import KTHDataset, get_dataloader
from src.models.model import build_model, build_original_model
from src.utils.criterion import MSE
from src.training.train_function import single_iter

if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("-n","--num_blocks", dest="num_blocks", type=int, default=6, help="Number of blocks in the model")
        parser.add_argument("-e","--epochs", dest="epochs", type=int, default=6, help="Number of epochs to train")
        parser.add_argument("-v","--val_per_epoch", dest="val_per_epoch", type=int, default=5, help="Number of epochs to validation")   
        parser.add_argument("-s","--seq_len", type=int, dest="seq_len", default=5, help="Number of attention heads")  
        parser.add_argument("-b","--batch_size", type=int, dest="batch_size", default=64, help="Number of samples in 1 batch")  
        args = parser.parse_args()

        attempt_time = datetime.now()
        set_seed(2023)
        #Avoid override checkpoints from previous attempt => Add time to the checkpoint folder name
        ckpt_save_dir = Path.cwd().joinpath("checkpoint/checkpoint_" + attempt_time.strftime("%Y-%m-%d_%H-%M-%S"))
        tensorboard_save_dir = Path.cwd().joinpath("tensorboard/tensorboard_" + attempt_time.strftime("%Y-%m-%d_%H-%M-%S"))
        resume_ckpt = None

        if not Path(ckpt_save_dir).exists():
                Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, 
                        datefmt="%a, %d %b %Y %H:%M:%S", 
                        format="[%(levelname)s] %(message)s - (%(filename)s)",      # e.g., [INFO] Log message 1 - (main.py)
                        filename=ckpt_save_dir.joinpath("train_log.log").absolute().as_posix(), 
                        filemode="a")

        start_epoch = 0
        summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())
        epochs = args.epochs
        val_per_epoch = args.val_per_epoch
        lr = 1e-4
        dropout = 0.1
        device = get_device_available()         # device = cuda/mps if available. Otherwise, device = cpu

        ##################### Init Dataset ###########################
        root_dir = "./data"
        seq_len = args.seq_len             # Number of previous frames
        batch_size = args.batch_size
        train_split = 0.7                    
        val_split = 0.15
        test_split = 0.15
        # Initialise the dataset
        full_dataset = KTHDataset(root_dir, seq_len)  
        # Get the dataloader
        train_loader, val_loader, test_loader = get_dataloader(full_dataset, batch_size, train_split, val_split, test_split)

        ##################### Init Model ###########################
        d_model = 768               # feature dimension of an input embedding
        N = args.num_blocks                       # Number of encoder blocks in the model
        h = 6                   # Number of heads
        #model = build_model(d_model, seq_len, N, h).to(device)
        original_model = build_original_model(d_model, seq_len, N, h).to(device)
        optimizer = torch.optim.AdamW(params=original_model.parameters(), lr=1e-4)
        #model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        original_model_num_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
        #print(f"Total number of model's parameters: {model_num_params}")
        print(f"Total number of original model's parameters: {original_model_num_params}")
        ##################### Init Loss Function ###########################
        loss_name_list = ["Total", "MSE", "SCL"]
        loss_dict = init_loss_dict(loss_name_list)
        original_loss_dict = init_loss_dict(loss_name_list)
        mse = MSE()

        ##################### Resume training from checkpoint ###########################
        '''if resume_ckpt is not None:
            # Load the checkpoint
            start_epoch, loss_dict, model_state_dict, optimizer_state_dict = load_ckpt(resume_ckpt, model, optimizer, loss_dict)
            # Load the model and the optimizer
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)'''
        
        ##################### Train (SCP) ###########################
        '''for epoch in range(start_epoch + 1, epochs +  1):
                loss_dict["epochs"] = epoch
                #Init temporary loss dict (Individual loss of each iteration)
                #Get current date/time
                epoch_time = datetime.now()
                epoch_loss_dict = init_loss_dict(loss_name_list)
                #Train phase
                iter_count = len(train_loader.dataset)
                for idx, batch in enumerate(train_loader, 0):
                        iter_loss_dict = single_iter(model, optimizer, batch, device, mse)
                        #Add loss to epoch loss
                        for k, v in iter_loss_dict.items():
                                epoch_loss_dict[k].train.append(v)
                #Take average epoch loss
                for k, v in epoch_loss_dict.items():
                        if (k != "epochs"):
                                loss_concat = torch.stack(v.train)
                                loss_dict[k].train.append(torch.mean(loss_concat))
                write_summary(summary_writer, loss_dict)

                if (epoch % val_per_epoch == 0):
                #Evaluation phase
                        for idx, batch in enumerate(val_loader, 0):
                                iter_loss_dict = single_iter(model, optimizer, batch, device, mse, train_flag = False)
                                #Add loss to epoch loss
                                for k, v in iter_loss_dict.items():
                                        epoch_loss_dict[k].val.append(iter_loss_dict[k])
                        #Take average epoch loss
                        for k, v in epoch_loss_dict.items():
                                if (k != "epochs"):
                                        loss_concat = torch.stack(v.val)
                                        loss_dict[k].val.append(torch.mean(loss_concat))
                        write_summary(summary_writer, loss_dict, train_flag=False)

                #Save checkpoint
                save_ckpt(model, optimizer, epoch, loss_dict, ckpt_save_dir)
                
                epoch_time_used = datetime.now() - epoch_time
                print(f"Estimated remaining training time: {epoch_time_used.total_seconds()/3600. * (start_epoch + epochs - epoch)} Hours")
                logging.info(f"epoch {epoch}, {epoch_loss_dict["Total"]}")
                logging.info(f"SCP model, Epoch {epoch} used {str(epoch_time_used)}")
                logging.info(f"Estimated remaining training time: {epoch_time_used.total_seconds()/3600. * (start_epoch + epochs - epoch)} Hours")'''

        ##################### Train (original) ###########################
        for epoch in range(start_epoch + 1, epochs +  1):
                original_loss_dict["epochs"] = epoch
                #Init temporary loss dict (Individual loss of each iteration)
                #Get current date/time
                epoch_time = datetime.now()
                epoch_loss_dict = init_loss_dict(loss_name_list)
                #Train phase
                iter_count = len(train_loader.dataset)
                for idx, batch in enumerate(train_loader, 0):
                        iter_loss_dict = single_iter(original_model, optimizer, batch, device, mse)
                        #Add loss to epoch loss
                        for k, v in iter_loss_dict.items():
                                epoch_loss_dict[k].train.append(v)
                #Take average epoch loss
                for k, v in epoch_loss_dict.items():
                        if (k != "epochs"):
                                loss_concat = torch.stack(v.train)
                                original_loss_dict[k].train.append(torch.mean(loss_concat))
                write_summary_comparison(summary_writer, original_loss_dict)

                #Save checkpoint
                save_ckpt(original_model, optimizer, epoch, loss_dict, ckpt_save_dir)

                if (epoch % val_per_epoch == 0):
                #Evaluation phase
                        for idx, batch in enumerate(val_loader, 0):
                                iter_loss_dict = single_iter(original_model, optimizer, batch, device, mse, train_flag = False)
                                #Add loss to epoch loss
                                for k, v in iter_loss_dict.items():
                                        epoch_loss_dict[k].val.append(iter_loss_dict[k])
                        #Take average epoch loss
                        for k, v in epoch_loss_dict.items():
                                if (k != "epochs"):
                                        loss_concat = torch.stack(v.val)
                                        original_loss_dict[k].val.append(torch.mean(loss_concat))
                        write_summary_comparison(summary_writer, original_loss_dict, train_flag=False)
                
                epoch_time_used = datetime.now() - epoch_time
                print(f"Estimated remaining training time: {epoch_time_used.total_seconds()/3600. * (start_epoch + epochs - epoch)} Hours")
                logging.info(f"epoch {epoch}, {epoch_loss_dict["Total"]}")
                logging.info(f"Original model, Epoch {epoch} used {str(epoch_time_used)}")
                logging.info(f"Estimated remaining training time: {epoch_time_used.total_seconds()/3600. * (start_epoch + epochs - epoch)} Hours")

        ##################### Test ###########################
        #Loss dictionary used for testing
        '''test_loss_dict = init_loss_dict(loss_name_list)
        test_iter_loss_dict = init_loss_dict(loss_name_list)
        for idx, batch in enumerate(test_loader, 0):
                print(idx)
                temp_loss_dict = single_iter(model, optimizer, batch, device, mse, train_flag=False)
                temp_loss_dict_comp = single_iter(original_model, optimizer, batch, device, mse, train_flag=False)
                for k, v in temp_loss_dict.items():
                        if k != 'epochs':
                                #train is for SCP model, val is for original model
                                test_iter_loss_dict[k].train.append(temp_loss_dict[k])
                                test_iter_loss_dict[k].val.append(temp_loss_dict_comp[k])
                
        for k, v in test_iter_loss_dict.items():
                if (k != 'epochs'):
                        loss_concat = torch.stack(v.train)
                        loss_concat_comp = torch.stack(v.val)
                        test_loss_dict[k].train.append(torch.mean(loss_concat))
                        test_loss_dict[k].val.append(torch.mean(loss_concat_comp))

        print(f"MSE loss of SCP model: {loss_dict['MSE'].train[0]}")
        print(f"MSE loss of original model: {loss_dict['MSE'].val[0]}")
        logging.info(f"MSE loss of SCP model: {loss_dict['MSE'].train[0]}")
        logging.info(f"MSE loss of original model: {loss_dict['MSE'].val[0]}")'''