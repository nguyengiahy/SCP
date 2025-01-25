import torch

def single_iter(model, optimizer, batch, device, task_loss, scl_weight=0.7, train_flag=True):
  '''
  Inputs:
  model: SCP model
  optimizer: Optimizer used
  batch: A single batch from loader
  device: Operating device (cuda/mps/cpu) -> Can be optimized by using get_available_device
  task_loss: Loss function for specific task
  scl_weight: Weight of SCL loss
  train_flag: True if training, False if evaluating
  '''
  #Get frames and label to the device
  past_frames, label = batch # past_frames (batch, seq_len, channel, width, height), label (batch, d_model)
  past_frames.to(device)
  label.to(device)

  #Train phase
  if train_flag:
    #Reset gradients of model
    model.zero_grad(set_to_none=True)
    #Forward pass
    pred_frame, scl_loss = model(past_frames) # pred_frame (batch, d_model (default=768)), scl_loss (batch, 1)

    if optimizer is not None:
      #All parameters need to update have been set to require_grad = True earlier => Skip this modify step
      task_loss_value = task_loss(pred_frame, label)
      #Apply loss formula (total loss = task_loss + scl_loss * lambda)
      total_loss = task_loss_value + torch.mul(scl_loss, scl_weight)
      total_loss.backward()
      optimizer.step()

  #Evaluate phase
  else:
    #Reset gradients of model
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
      #Forward pass
      pred_frame, scl_loss = model(past_frames)
      #All parameters need to update have been set to require_grad = True earlier => Skip this modify step
      task_loss_value = task_loss(pred_frame, label)
      #Apply loss formula (total loss = task_loss + scl_loss * lambda)
      total_loss = task_loss_value + torch.mul(scl_loss, scl_weight)

  past_frames.to('cpu')
  label.to('cpu')
  iter_loss_dict = {'Total': total_loss, 'MSE': task_loss_value, 'SCL': scl_loss}
  del total_loss
  return iter_loss_dict
