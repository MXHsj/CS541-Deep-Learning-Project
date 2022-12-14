# ===============================================================
# file name:    model.py
# description:  naive U-Net implementation
# author:
# date:         2022-11-12
# ===============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
# for resnet
from torchvision import models

# =========== ResNet ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model = models.resnet18(weights=None)
base_model = base_model.to(device)
# =========================================

# ===============================================================
# file name:    model.py
# description:  naive U-Net implementation
# author:
# date:         2022-11-12
# ===============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super().__init__()
    if not mid_channels:
      mid_channels = out_channels
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)

def root_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), 
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), 
    )

class Up(nn.Module):
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels, bilinear=True):
    super().__init__()

    # if bilinear, use the normal convolutions to reduce the number of channels
    if bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    else:
      self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
      self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = nn.functional.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
    #x1 = self.up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x1, x2], dim=1)
    return self.conv(x)
    
class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels, encoder=True):
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    self.encoder = encoder

  def forward(self, x):
    # return self.conv(x)
    if not self.encoder:
      return F.softmax(self.conv(x), dim=1)  # normalize class probabilities
    else:
      return self.conv(x)

# Define the UNet architecture
class ResNetUNet(nn.Module):

    def __init__(self, n_channels, n_classes, encoder=True, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.dconv_down1 = DoubleConv(n_channels, 64)
        self.dconv_down11 = root_block(64, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down21 = root_block(128, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down31 = root_block(256, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.dconv_down41 = root_block(512, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self.dconv_up3 = Up(256 + 512, 256)
        self.dconv_up31 = root_block(256, 256)
        self.dconv_up2 = Up(128 + 256, 128)
        self.dconv_up21 = root_block(128, 128)
        self.dconv_up1 = Up(128 + 64, 64)
        self.dconv_up11 = root_block(64, 64)

        self.conv_last = OutConv(64, n_classes, encoder=encoder)

    def forward(self, x):
        #print(f"input shape: {x.shape}")
        conv1 = self.dconv_down1(x)
        #print(f"conv1 shape: {conv1.shape}")
        x = self.dconv_down11(conv1)
        #print(f"dconv_down11 shape: {x.shape}")
        x += conv1
        #print(f"x shape: {x.shape}")
        x = self.relu(x)
        x = self.maxpool(x)
        #print(f"x shape: {x.shape}")

        conv2 = self.dconv_down2(x)
        #print(f"conv2 shape: {conv2.shape}")
        x = self.dconv_down21(conv2)
        #print(f"dconv_down21 shape: {x.shape}")
        x += conv2
        #print(f"x shape: {x.shape}")
        x = self.relu(x)
        x = self.maxpool(x)
        #print(f"x shape: {x.shape}")

        conv3 = self.dconv_down3(x)
        #print(f"conv3 shape: {conv3.shape}")
        x = self.dconv_down31(conv3)
        #print(f"dconv_down31 shape: {x.shape}")
        x += conv3
        #print(f"x shape: {x.shape}")
        x = self.relu(x)
        x = self.maxpool(x)
        #print(f"x shape: {x.shape}")

        conv4 = self.dconv_down4(x)
        #print(f"conv4 shape: {conv4.shape}")
        x = self.dconv_down41(conv4)
        #print(f"dconv_down41 shape: {x.shape}")
        x += conv4
        #print(f"x shape: {x.shape}")
        x = self.relu(x)
        #print(f"x shape: {x.shape}")
        #print("----------------------------------------------------")
        deconv3 = self.dconv_up3(x, conv3)
        #deconv3 = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #print(f"deconv3 shape: {deconv3.shape}")
        #deconv3 = torch.cat([deconv3, conv3], dim=1)
        #print(f"deconv3 shape: {deconv3.shape}")
        #uconv3 = self.dconv_up3(deconv3)
        x = self.dconv_up31(deconv3)
        #print(f"x shape: {x.shape}")
        x += deconv3
        #print(f"x shape: {x.shape}")
        x = self.relu(x)
        #print(f"x shape: {x.shape}")

        #print("----------------------------------------------------")
        deconv2 = self.dconv_up2(x, conv2)
        #print(f"deconv2 shape: {deconv2.shape}")
        #deconv2 = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #deconv2 = torch.cat([deconv2, conv2], dim=1)
        #uconv2 = self.dconv_up2(deconv2)
        x = self.dconv_up21(deconv2)
        #print(f"x shape: {x.shape}")
        x += deconv2
        #print(f"x shape: {x.shape}")
        x = self.relu(x)
        #print(f"x shape: {x.shape}")

        #print("----------------------------------------------------")
        deconv1 = self.dconv_up1(x, conv1)
        #print(f"deconv1 shape: {deconv1.shape}")
        #deconv1 = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #deconv1 = torch.cat([deconv1, conv1], dim=1)
        #uconv1 = self.dconv_up1(deconv1)
        x = self.dconv_up11(deconv1)
        #print(f"x shape: {x.shape}")
        x += deconv1
        #print(f"x shape: {x.shape}")
        x = self.relu(x)
        #print(f"x shape: {x.shape}")

        out = self.conv_last(x)

        return out