import torch
import argparse
import numpy as np
from torch import nn
import albumentations as A
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from .Code.datasets.uavid import UAVDataset
from .Code.models.sam import SegmentationVITSAM
from .Code.trainers.uav_segmentation import UAVSegmentationTrainer

def get_configuration():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description='Start an experiement with given configuration file.')
  
  parser.add_argument("--config", type=str, help="Path to a config file", required=True)

  option = parser.parse_args()
  config = option.config
  return config

if __name__ == "__main__":

  config = get_configuration()

  # Setup model
  model = SegmentationVITSAM(embed_dim=config.model.embed_dim,
                             num_heads=config.model.num_heads,
                             depth=config.model.depth,
                             extract_layers=config.model.extract_layers,
                             encoder_global_attn_indexes=config.model.encoder_global_attn_indexes,
                             drop_rate=config.model.drop_rate,
                             num_classes=config.model.num_classes,
                             ckpt_path=config.model.ckpt_path)
  
  for p in model.encoder.parameters():
    p.requires_grad = False

  
  # Setup Transform
  p = config.transform.apply_augmentation_ratio
  input_shape = config.transform.input_shape

  train_transform = A.Compose([
            A.RandomResizedCrop(
              height=input_shape,
              width=input_shape,
              scale=(0.5, 1)
              ),
            A.RandomRotate90(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            ToTensorV2()
      ])

  val_transform = A.Compose([A.Resize(height=input_shape, width=input_shape), ToTensorV2()])


  # Setup Dataset
  train_data = UAVDataset(config.training.train_dir, transform=train_transform)
  val_data = UAVDataset(config.training.val_dir, transform=val_transform)
  train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.training.batch_size)
  val_dataloader = DataLoader(val_data, shuffle=False, batch_size=config.training.batch_size)

  # Setup Training
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), betas=(0.85, 0.95), weight_decay=0.0001, lr=config.training.lr)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  mixed_precision = config.training.mixed_precision

  # Setup trainer
  trainer = UAVSegmentationTrainer(model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   device=device,
                                   mixed_precision=mixed_precision)
  
  # Training
  trainer.fit(epochs=config.training.epochs,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            eval_every=config.training.eval_every)