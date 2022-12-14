import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class BottleneckLayerTheta(nn.Module):
  def __init__(self, in_channels_num, growth_rate):
    super().__init__()
    self.bottle_neck = nn.Sequential(
      nn.BatchNorm2d(in_channels_num),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels_num, 4*growth_rate, kernel_size=(1,1), bias=False),
      #nn.Dropout(0.2),

      nn.BatchNorm2d(4*growth_rate),
      nn.ReLU(inplace=True),
      nn.Conv2d(4*growth_rate, growth_rate, kernel_size=(3,3), padding=1, bias=False)
      #nn.Dropout(0.2)
    )

  def forward(self, x):
    output = self.bottle_neck(x)
    return torch.cat([x, output], 1)
class BottleneckLayerThetaLayer(nn.Module):
  def __init__(self, in_channels_num, growth_rate, out_channels, mid_mult):
    super().__init__()
    self.bottle_neck = nn.Sequential(
      nn.BatchNorm2d(in_channels_num),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels_num, mid_mult*growth_rate, kernel_size=(1,1), bias=False),
      #nn.Dropout(0.2),

      nn.BatchNorm2d(mid_mult*growth_rate),
      nn.ReLU(inplace=True),
      nn.Conv2d(mid_mult*growth_rate, out_channels, kernel_size=(3,3), padding=1, bias=False)
      #nn.Dropout(0.2)
    )

  def forward(self, x):
    output = self.bottle_neck(x)
    return output
class TransitionLayer(nn.Module):
  def __init__(self, in_channels_num, out_channels_num):
    super().__init__()

    self.transition = nn.Sequential(
      nn.BatchNorm2d(in_channels_num),
      nn.Conv2d(in_channels_num, out_channels_num, (1,1), bias=False),
      #nn.Dropout(0.2),
      nn.AvgPool2d((2,2), stride=2)
    )

  def forward(self, x):
    x = self.transition(x)
    return x
class ThetaBlock(nn.Module):
  def get_connects(self, layer):
    thresh = int(self.angle * layer)
    cons = []
    for i in range(layer):
      if i >= thresh or (i==0):
        cons.append(i)
    channels = len(cons) * self.growth_rate
    channels = 0
    for i in cons:
      channels += self.layer_ins[i]

    return cons, channels

  def __init__(self, in_channels_num, growth_rate, nlayers, angle, keepbase, mid_mult):
    super().__init__()
    self.layers_ = []
    self.angle = angle
    self.keepBase = keepbase
    self.growth_rate = growth_rate
    self.start = in_channels_num
    self.mid_mult = mid_mult
    self.layer_ins = []
    self.layer_ins.append(in_channels_num)
    for l in range(nlayers):

      in_channels_num = self.get_connects(l+1)[1]
      out_chans = growth_rate
      
      out_chans = int(out_chans)

      self.layers_.append(BottleneckLayerThetaLayer(in_channels_num, growth_rate, out_chans, self.mid_mult))
      self.layer_ins.append(out_chans)

    self.layers = nn.ModuleList(self.layers_)

  def get_out_channels(self):
    layer = len(self.layers_)
    cons, chans = self.get_connects(layer+1)
    if 0 not in cons:
      chans += self.start
    return chans

  def forward(self, x):
    layers_ = [x]
        
    for layer in range(len(self.layers)):
        link = self.get_connects(layer+1)[0]

        tin = []
        for i in link:
            tin.append(layers_[i])
        if len(tin) > 1:            
            x = torch.cat(tin, 1)
        else:
            x = tin[0]
        out = self.layers[layer](x)
        layers_.append(out)
        
    t = len(layers_)
    thresh = int(self.angle * (t))
    out_ = []
    for i in range(t):
      if (i == 0 and self.keepBase) or \
          i >= thresh or i==0:
          out_.append(layers_[i])
    out = torch.cat(out_, 1)
    return out


class DenseNetTheta(nn.Module):
  def __init__(self, growth_rate, num_blocks, theta, mid_mult):
    super().__init__()
    cur_channels = 2*growth_rate
    self.angle = theta
    self.network = nn.Sequential()
    self.network.append(nn.Conv2d(3, cur_channels, kernel_size=(3,3), padding=1, bias=False))
    
    for nb in num_blocks[0:-1]:
      bnt = ThetaBlock(cur_channels, growth_rate, nb, self.angle, False, mid_mult)
      self.network.append(bnt)
      cur_channels = bnt.get_out_channels()
      out_channels = int(cur_channels // 2)

      self.network.append(TransitionLayer(cur_channels, out_channels))
      
      cur_channels = out_channels
    bnt = ThetaBlock(cur_channels, growth_rate, num_blocks[-1], self.angle, False, mid_mult)
    self.network.append(bnt)
    cur_channels = bnt.get_out_channels()
    self.network.append(nn.BatchNorm2d(cur_channels))
    self.network.append(nn.ReLU(inplace=True))

    self.pool = nn.AdaptiveAvgPool2d((1,1))

    self.linear = nn.Linear(cur_channels, 100, bias=True)

  def forward(self, x):
    x = self.network(x)
    x = self.pool(x)
    x = x.view(x.size()[0], -1)
    x = self.linear(x)
    return x