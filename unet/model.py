# ===============================================================
# file name:    model.py
# description:  naive U-Net implementation
# author:
# date:         2022-11-12
# ===============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)

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
    x1 = self.up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

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

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # input is CHW
        diffY = x.size()[2] - g.size()[2]
        diffX = x.size()[3] - g.size()[3]

        g1 = F.pad(g1, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, g1], dim=1)
        
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi, x

class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, encoder=True, bilinear=False):
        super(AttentionUNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Maxpool = nn.MaxPool2d(2)

        self.Conv1 = DoubleConv(in_channels=n_channels,out_channels=64)
        self.Conv2 = DoubleConv(in_channels=64,out_channels=128)
        self.Conv3 = DoubleConv(in_channels=128,out_channels=256)
        self.Conv4 = DoubleConv(in_channels=256,out_channels=512)
        self.Conv5 = DoubleConv(in_channels=512,out_channels=1024)

        self.Up5 = up_conv(in_channels=1024,out_channels=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = Up(in_channels=1024, out_channels=512)

        self.Up4 = up_conv(in_channels=512,out_channels=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = Up(in_channels=512, out_channels=256)
        
        self.Up3 = up_conv(in_channels=256,out_channels=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = Up(in_channels=256, out_channels=128)
        
        self.Up2 = up_conv(in_channels=128,out_channels=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = Up(in_channels=128, out_channels=64)

        self.Conv_1x1 = OutConv(64, n_classes, encoder=encoder)


    def forward(self,x):
        #print(f"input shape: {x.shape}")
        # encoding path
        x1 = self.Conv1(x)
        #print(f"x1 shape: {x1.shape}")
        x2 = self.Maxpool(x1)
        #print(f"x2_1 shape: {x2.shape}")
        x2 = self.Conv2(x2)
        #print(f"x2_2 shape: {x2.shape}")
        
        x3 = self.Maxpool(x2)
        #print(f"x3_1 shape: {x3.shape}")
        x3 = self.Conv3(x3)
        #print(f"x3_2 shape: {x3.shape}")

        x4 = self.Maxpool(x3)
        #print(f"x4_1 shape: {x4.shape}")
        x4 = self.Conv4(x4)
        #print(f"x4_1 shape: {x4.shape}")

        x5 = self.Maxpool(x4)
        #print(f"x5_1 shape: {x5.shape}")
        x5 = self.Conv5(x5)
        #print(f"x5_1 shape: {x5.shape}")

        # decoding + concat path
        d5 = self.Up5(x5)
        #print(f"d5 shape: {d5.shape}")
        x4, d5 = self.Att5(g=d5,x=x4)
        #print(f"x4 shape: {x4.shape}")
        #d5 = torch.cat((x4,d5),dim=1)   
        #print(f"d5 shape: {d5.shape}")     
        d5 = self.Up_conv5(d5, x4)
        #print(f"d5 shape: {d5.shape}")
        
        d4 = self.Up4(d5)
        x3, d4 = self.Att4(g=d4,x=x3)
        #d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4, x3)

        d3 = self.Up3(d4)
        x2, d3 = self.Att3(g=d3,x=x2)
        #d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3, x2)

        d2 = self.Up2(d3)
        x1, d2 = self.Att2(g=d2,x=x1)
        #d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2, x1)

        d1 = self.Conv_1x1(d2)

        return d1

