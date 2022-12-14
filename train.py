# =======================================================================
# file name:    train.py
# description:  train network
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-13
# version:
# =======================================================================
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset

from unet.model import UNet
from utils.data_loader import LUSDataset, RandomGenerator
from utils.dice_score import dice_loss
from utils.evaluate import evaluate_dice
from utils.vis import tensor2array, plot_segmentation

dir_checkpoint = Path('./checkpoints/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              amp: bool = False,
              encoder: bool = True):
  # ========== Create dataset & split into train / validation partitions ==========
  random_generator = RandomGenerator()  # for data augmentation
  dataset = LUSDataset(encoder=encoder, transform=random_generator)
  n_val = int(len(dataset) * val_percent)
  n_train_all = len(dataset) - n_val
  train_set_all, val_set = random_split(dataset, [n_train_all, n_val], generator=torch.Generator().manual_seed(0))

  # ********** for experiment purpose **********
  train_set = Subset(train_set_all, torch.arange(50))  # take partial training data
  n_train = len(train_set)
  # ********************************************

  loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
  train_loader = DataLoader(train_set, shuffle=True, **loader_args)
  val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

  print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}''')

  # ========== Set up the optimizer, loss, learning rate scheduler, k-fold ==========
  # L2_reg = 1e-6  # 1e-8
  # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=L2_reg, momentum=0.9)
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.4)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
  grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
  criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 0., 0.]).to(device=device))  # supervise only the background
  global_step = 0

  val_score_rec = []
  loss_dice_rec = []
  loss_ce_rec = []
  loss_mse_rec = []
  val_max = 0.

  # ========== Begin training ==========
  for epoch in range(1, epochs + 1):
    net.train()
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
      for batch in train_loader:
        images = batch['image']
        masks_true = batch['mask']

        assert images.shape[1] == net.n_channels, \
            f'Network has been defined with {net.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        images = images.to(device=device)
        masks_true = masks_true.to(device=device)

        with torch.cuda.amp.autocast(enabled=amp):
          masks_pred = net(images)
          # print(f"mask shape: {masks_true.shape}, max val: {torch.max(masks_true[0])}")
          # print(f"pred mask shape: {masks_pred.shape}, max val: {torch.max(masks_pred[0,0,:,:])}")
          if not encoder:
            loss_CE = criterion(masks_pred, masks_true)
            loss_dice = dice_loss(masks_pred.float(),
                                  F.one_hot(masks_true, net.n_classes).permute(0, 3, 1, 2).float(),
                                  multiclass=True)
            loss = 0.1*loss_CE + loss_dice
            pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_ce (batch)': loss_CE.item(), 'loss_dice (batch)': loss_dice.item()})
          else:
            loss_fn = nn.MSELoss()
            loss = loss_fn(masks_pred.float(), masks_true.float())
            pbar.set_postfix(**{'loss (batch)': loss.item()})

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        pbar.update(images.shape[0])
        global_step += 1

        # ===== Evaluation round =====
        division_step = (n_train // (10 * batch_size))
        if division_step > 0:
          if global_step % division_step == 0:
            if not encoder:
              val_score = evaluate_dice(net, val_loader, device)
              scheduler.step(val_score)
              print(f"Validation Dice score: {val_score}")
              print(f'''Validation info:
                                Learning rate: {optimizer.param_groups[0]['lr']}
                                Validation Dice: {val_score}
                                Step: {global_step}
                                Epoch: {epoch}''')
            else:
              pass
            # ====== save figures ======
            tag = 'epoch_' + str(epoch) + '_step_' + str(global_step)
            image = tensor2array(images[0].float())
            if encoder:
              mask_true = tensor2array(masks_true[0].float())
              mask_pred = tensor2array(masks_pred[0].float())
              loss_mse_rec.append(loss.item())
            else:
              mask_true = tensor2array(masks_true[0].float())
              mask_pred = tensor2array(masks_pred.argmax(dim=1)[0].float())
              loss_ce_rec.append(loss_CE.item())
              loss_dice_rec.append(loss_dice.item())
              val_score_rec.append(val_score.item())
              val_max = val_score.item() if val_score.item() > val_max else val_max
            plot_segmentation(tag, image, mask_true, mask_pred)
            # ==========================

    if save_checkpoint:
      Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
      torch.save(net.state_dict(), './checkpoints/' + str(encoder) + f'checkpoint_epoch{epoch}.pth')
      print(f"Checkpoint {epoch} saved!")

  fig = plt.figure(figsize=(10, 7))
  if encoder:
    plt.plot(np.arange(len(loss_mse_rec)), loss_mse_rec, label='mse loss')
  else:
    plt.plot(np.arange(len(val_score_rec)), val_score_rec, label='dice score')
    plt.plot(np.arange(len(loss_dice_rec)), loss_dice_rec, label="dice loss")
  plt.legend()
  plt.savefig('training_log/loss_curve.png')
  plt.close(fig)
  print(f'best validation accuracy: {val_max}')


def get_args():
  parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
  parser.add_argument('--epochs', '-e', metavar='E', type=int, default=75, help='Number of epochs')
  parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
  parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate', dest='lr')  # 2e-4
  parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
  parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
  parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                      help='Percent of the data that is used as validation (0-100)')
  parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
  parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
  parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')

  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device {device}")

  isTrainEncoder = False   # set to true if training encoder only
  if isTrainEncoder:
    print('------------------------------train encoder---------------------------------')
    net_encoder = UNet(n_channels=1, n_classes=1, encoder=True, bilinear=args.bilinear)
    print(f'''Network:\n
      \t{net_encoder.n_channels} input channels\n
      \t{net_encoder.n_classes} output channels (classes)\n
      \t{"Bilinear" if net_encoder.bilinear else "Transposed conv"} upscaling''')
    net_encoder.to(device=device)
    try:
      train_net(net=net_encoder,
                epochs=6,  # args.epochs
                batch_size=4,
                learning_rate=1e-3,
                device=device,
                val_percent=10/100,
                amp=args.amp,
                encoder=True)
    except KeyboardInterrupt:
      torch.save(net_encoder.state_dict(), str(dir_checkpoint / 'INTERRUPTED.pth'))
      print(f"Saved interrupt")
      raise
  else:
    print('------------------------------train decoder---------------------------------')
    net_encoder = torch.load(os.path.dirname(__file__) + '/checkpoints/Truecheckpoint_epoch5.pth')
    net_decoder = UNet(n_channels=1, n_classes=args.classes, encoder=False, bilinear=args.bilinear)
    print(f'''Network:\n
      \t{net_decoder.n_channels} input channels\n
      \t{net_decoder.n_classes} output channels (classes)\n
      \t{"Bilinear" if net_decoder.bilinear else "Transposed conv"} upscaling''')
    net_decoder.to(device=device)   # force tensors on same device

    for i, (name, parameters) in enumerate(net_decoder.named_parameters()):
      if i < 30:
        # parameters.requires_grad = False
        parameters.data = net_encoder[name]  # was 'net_encoder.parameters()[name]'
    try:
      train_net(net=net_decoder,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                val_percent=args.val / 100,
                amp=args.amp,
                encoder=False)
    except KeyboardInterrupt:
      torch.save(net_decoder.state_dict(), str(dir_checkpoint / 'INTERRUPTED.pth'))
      print(f"Saved interrupt")
      raise
