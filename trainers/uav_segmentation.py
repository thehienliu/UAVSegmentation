import os
import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Union
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

class UAVSegmentationTrainer:
  def __init__(self,
               model: nn.Module,
               criterion: nn.Module,
               optimizer: Optimizer,
               scheduler: _LRScheduler,
               device: Union[str, torch.device]='cpu',
               mixed_precision: bool=False):

    self.model = model
    self.device = device
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.mixed_precision = mixed_precision
    if self.mixed_precision:
      self.scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
      self.scaler = None

  def fit(self,
          epochs: int,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          eval_every: int = 1,
          continue_epoch: int = 0):
    """Fitting function to start training and validation of the trainer"""

    self.model.to(self.device)

    for epoch in range(epochs):
      print(f"Epoch: {epoch + 1}")
      
      ######### Training #########
      train_metrics = self.train_epoch(train_dataloader)
      print('Train epoch stats: ')
      print(train_metrics)

      ######### Evaluating  #########
      if (epoch + 1) % eval_every == 0:
        val_metrics = self.eval_epoch(val_dataloader)
        print('Evaluate epoch stats: ')
        print(val_metrics)

        states = {
          'epoch': continue_epoch + epoch + 1,
          "model_state_dict": self.model.state_dict(),
          "scheduler_state_dict": self.scheduler.state_dict(),
        }

        self.save_checkpoint(continue_epoch + epoch + 1, states, 'checkpoint')
        self.save_checkpoint(continue_epoch + epoch + 1, train_metrics, 'train_metrics', metric=True)
        self.save_checkpoint(continue_epoch + epoch + 1, val_metrics, 'val_metrics', metric=True)

      # Log learning rate
      curr_lr = self.optimizer.param_groups[0]['lr']

      if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
        self.scheduler.step(float(val_metrics["loss"]))
      else:
        self.scheduler.step()

      new_lr = self.optimizer.param_groups[0]['lr']
      print(f'Old lr: {curr_lr:.6f} - New lr: {new_lr:.6f}')

  def train_epoch(self, train_dataloader: DataLoader):
    """Training logic for a training epoch"""

    epoch_metrics = {'loss': [], 'accuracy': []}

    self.model.train()
    for batch in tqdm(train_dataloader):

        batch_metrics = self.step(batch, train=True)

        # Save metrics
        for k, v in batch_metrics.items():
          epoch_metrics[k].append(v)

    return self.mean_epoch_metrics(epoch_metrics)

  def eval_epoch(self, val_dataloader: DataLoader):
    """Evaluate logic for a val epoch"""

    epoch_metrics = {'loss': [], 'accuracy': []}

    self.model.eval()
    with torch.no_grad():
      for batch in tqdm(val_dataloader):

        batch_metrics = self.step(batch, train=False)

        # Save metrics
        for k, v in batch_metrics.items():
          epoch_metrics[k].append(v)

    return self.mean_epoch_metrics(epoch_metrics)

  def mean_epoch_metrics(self, epoch_metrics):
    return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

  def step(self, batch, train=True):
    images, labels = batch['image'].to(self.device), batch['label'].to(self.device)

    if self.mixed_precision:
      with torch.autocast(device_type="cuda", dtype=torch.float16):

        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        if train:
          # Backward pass
          self.optimizer.zero_grad()
          self.scaler.scale(loss).backward()
          self.scaler.step(self.optimizer)
          self.scaler.update()
          self.model.zero_grad()

    else:
      # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        if train:
        # Backward pass
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          self.model.zero_grad()

    batch_metrics = {}
    batch_metrics['loss'] = loss.item()
    batch_metrics['accuracy'] = (outputs.argmax(dim=1).view(-1) == labels.view(-1)).sum().item() / len(labels.view(-1))

    return batch_metrics

  def save_checkpoint(self, epoch: int, state: dict, name: str, metric: bool = False):
      checkpoint_dir = "metrics" if metric else "checkpoints"
      checkpoint_name = f"{name}_{epoch}.pth"
      os.makedirs(checkpoint_dir, exist_ok=True) # if dir already exist function does nothing

      filename = os.path.join(checkpoint_dir, checkpoint_name)
      torch.save(state, filename)