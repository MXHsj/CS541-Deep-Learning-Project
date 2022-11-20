# =======================================================================
# file name:    train.py
# description:  train network
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-13
# version:
# =======================================================================
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from unet.model import UNet
from utils.data_loader import LUSDataset
from utils.dice_score import dice_loss
from utils.evaluate import evaluate_dice
from utils.vis import tensor2PIL, plot_segmentation

dir_checkpoint = Path('./checkpoints/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              #img_scale: float = 0.5,
              amp: bool = False):
  # 1. Create dataset
  dataset = LUSDataset()

  # 2. Split into train / validation partitions
  n_val = int(len(dataset) * val_percent)
  n_train = len(dataset) - n_val
  train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

  # 3. Create data loaders
  loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
  train_loader = DataLoader(train_set, shuffle=True, **loader_args)
  val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

  # (Initialize logging)
  print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

  # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
  optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
  grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
  criterion = nn.CrossEntropyLoss()
  global_step = 0

  # 5. Begin training
  for epoch in range(1, epochs+1):
    net.train()
    epoch_loss = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
      for batch in train_loader:
        images = batch['image']
        masks_true = batch['mask']
        print(f"image shape: {images.shape}")

        assert images.shape[1] == net.n_channels, \
            f'Network has been defined with {net.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        images = images.to(device=device, dtype=torch.float32)
        masks_true = masks_true.to(device=device, dtype=torch.long)

        with torch.cuda.amp.autocast(enabled=amp):
          masks_pred = net(images)
          #print(f"type of images: {type(images)}")
          #print(f"size of images: {images.size()}")
          #print(f"type of masks_true: {type(masks_true)}")
          #print(f"size of masks_true: {masks_true.size()}")
          #print(f"shape of masks_pred: {masks_pred.shape}")
          #print(f"type of mask: {type(masks_true[0])}")
          print("loss part -----------------------")
          print(f"mask shape: {masks_true.shape}, max val: {torch.max(masks_true[0])}")
          #print(f"type of pred mask: {type(masks_pred[0])}")
          print(f"pred mask shape: {masks_pred.shape}, max val: {torch.max(masks_pred[0])}")

          loss = criterion(masks_pred, masks_true) \
              + dice_loss(F.softmax(masks_pred, dim=1).float(),
                          F.one_hot(masks_true, net.n_classes).permute(0, 3, 1, 2).float(),
                          multiclass=True)

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        pbar.update(images.shape[0])
        global_step += 1
        epoch_loss += loss.item()

        print(f"train loss: {loss.item()}")
        print(f"step: {global_step}")
        print(f"epoch: {epoch}")
        pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
        division_step = (n_train // (10 * batch_size))
        if division_step > 0:
          if global_step % division_step == 0:
            val_score = evaluate_dice(net, val_loader, device)
            scheduler.step(val_score)

            print(f"Validation Dice score: {val_score}")
            print(f'''Validation info:
                  Learning rate: {optimizer.param_groups[0]['lr']}
                  Validation Dice: {val_score}
                  Step: {global_step}
                  Epoch: {epoch}
            ''')

            # ====== showing figures will cause thread issue ======
            image = tensor2PIL(images[0].float())
            mask_true = tensor2PIL(masks_true[0].float())
            mask_pred = tensor2PIL(masks_pred.argmax(dim=1)[0].float())
            plot_segmentation(epoch, global_step, image, mask_true, mask_pred)
            # =====================================================

    if save_checkpoint:
      Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
      torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
      print(f"Checkpoint {epoch} saved!")


def get_args():
  parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
  parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
  parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
  parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                      help='Learning rate', dest='lr')
  parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
  parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
  parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                      help='Percent of the data that is used as validation (0-100)')
  parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
  parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
  parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')

  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device {device}")

  # Change here to adapt to your data
  # n_channels=3 for RGB images
  # n_channels=1 for grey images
  # n_classes is the number of probabilities you want to get per pixel
  net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

  print(f'''Network:\n
        \t{net.n_channels} input channels\n
        \t{net.n_classes} output channels (classes)\n
        \t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling
  ''')

  if args.load:
    net.load_state_dict(torch.load(args.load, map_location=device))
    print(f"Model loaded from {args.load}")
    #logging.info(f'Model loaded from {args.load}')

  net.to(device=device)
  try:
    train_net(net=net,
              epochs=args.epochs,
              batch_size=args.batch_size,
              learning_rate=args.lr,
              device=device,
              # img_scale=args.scale,
              val_percent=args.val / 100,
              amp=args.amp)
  except KeyboardInterrupt:
    torch.save(net.state_dict(), str(dir_checkpoint/'INTERRUPTED.pth'))
    print(f"Saved interrupt")
    #logging.info('Saved interrupt')
    raise

  finally:
    pass
