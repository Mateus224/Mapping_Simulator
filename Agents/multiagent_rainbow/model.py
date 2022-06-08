# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
import functools
import operator
from torch.nn.utils import spectral_norm


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.3):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(

                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)#,
                #nn.BatchNorm2d(out_channels)

            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)

        #self.bn1 = nn.BatchNorm2d(out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.conv1(input))
        input = nn.ReLU()(self.conv2(input))
        input = input + shortcut
        return nn.ReLU()(input)


class DQN_ResNet(nn.Module):
  def __init__(self, args, action_space, resblock, repeat):
    super(DQN_ResNet, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    filters = [128, 128, 256, 512, 1024]
    self.layer0 = nn.Sequential(
      nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1),
      #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      #nn.BatchNorm2d(64),

      nn.ReLU())

    self.layer1 = nn.Sequential()
    self.layer1.add_module('conv2_1', ResBlock(filters[0], filters[1], downsample=False))
    for i in range(1, repeat[0]):
            self.layer1.add_module('conv2_%d'%(i+1,), ResBlock(filters[1], filters[1], downsample=False))

    self.layer2 = nn.Sequential()

    self.layer2.add_module('conv3_1', ResBlock(filters[1], filters[2], downsample=True))

    for i in range(1, repeat[1]):
            self.layer2.add_module('conv3_%d' % (
                i+1,), ResBlock(filters[2], filters[2], downsample=False))


    #self.layer3 = nn.Sequential()
    #self.layer3.add_module('conv4_1', ResBlock(filters[2], filters[3], downsample=True))
    #for i in range(1, repeat[2]):
    #    self.layer3.add_module('conv4_%d' % (
    #        i+1,), ResBlock(filters[3], filters[3], downsample=False))

    #self.layer4 = nn.Sequential()
    #self.layer4.add_module('conv5_1', ResBlock(filters[3], filters[4], downsample=True))
    #for i in range(1, repeat[3]):
    #    self.layer4.add_module('conv5_%d'%(i+1,),ResBlock(filters[4], filters[4], downsample=False))

    self.dense = nn.Sequential(spectral_norm(nn.Linear(16384, 1024)), nn.ReLU())
    self.fc_h_v = NoisyLinear(1024, 512, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(1024, 512, std_init=args.noisy_std)

    self.fc_z_v = NoisyLinear(512, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(512, action_space * self.atoms, std_init=args.noisy_std) 

  def forward(self, x, log=False):
    input = self.layer0(x)
    input = self.layer1(input)
    input = self.layer2(input)

    #input = self.layer3(input)
    #input = self.layer4(input)
    input = torch.flatten(input, start_dim=1)
    input = self.dense(input)

    v_uuv = self.fc_z_v(F.relu(self.fc_h_v(input)))  # Value stream
    a_uuv = self.fc_z_a(F.relu(self.fc_h_a(input)))  # Advantage stream

    #v_uav, a_uav = v_uav.view(-1, 1, self.atoms), a_uav.view(-1, self.action_space, self.atoms)    
    v_uuv, a_uuv = v_uuv.view(-1, 1, self.atoms), a_uuv.view(-1, self.action_space, self.atoms)
    
    #q_uav = v_uav + a_uav - a_uav.mean(1, keepdim=True)  # Combine streams
    q_uuv = v_uuv + a_uuv - a_uuv.mean(1, keepdim=True)  # Combine streams

    if log:  # Use log softmax for numerical stability
      #q_uav = F.log_softmax(q_uav, dim=2)  # Log probabilities with action over second dimension
      q_uuv = F.log_softmax(q_uuv, dim=2)  # Log probabilities with action over second dimension
    else:
      #q_uav = F.softmax(q_uav, dim=2)  # Probabilities with action over second dimension
      q_uuv = F.softmax(q_uuv, dim=2)  # Probabilities with action over second dimension
    return  q_uuv #q_uav,
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc_z' in name:
        module.reset_noise()


class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space
    self.convs = nn.Sequential(nn.Conv2d(2, 128, 4, stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(128, 128, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Conv2d(128, 256, 2, stride=1, padding=0), nn.ReLU(), nn.Flatten())
    self.num_features_before_fcnn = functools.reduce(operator.mul, list(self.convs(torch.rand(1, *(2,16,16))).shape))
    #self.conv_output_size = 3136
    #self.dense = nn.Sequential(nn.Linear(self.num_features_before_fcnn, 512), nn.ReLU())
    self.fc_h_v = NoisyLinear(self.num_features_before_fcnn, 512, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.num_features_before_fcnn, 512, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(512, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(512, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x = self.convs(x)
    #x = x.view(-1, self.num_features_before_fcnn)
    #x = self.dense(x)
    #x = x.view(-1, self.conv_output_size)
    #v_uav = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    #a_uav = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream

    v_uuv = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a_uuv = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream

    #v_uav, a_uav = v_uav.view(-1, 1, self.atoms), a_uav.view(-1, self.action_space, self.atoms)    
    v_uuv, a_uuv = v_uuv.view(-1, 1, self.atoms), a_uuv.view(-1, self.action_space, self.atoms)
    
    #q_uav = v_uav + a_uav - a_uav.mean(1, keepdim=True)  # Combine streams
    q_uuv = v_uuv + a_uuv - a_uuv.mean(1, keepdim=True)  # Combine streams

    if log:  # Use log softmax for numerical stability
      #q_uav = F.log_softmax(q_uav, dim=2)  # Log probabilities with action over second dimension
      q_uuv = F.log_softmax(q_uuv, dim=2)  # Log probabilities with action over second dimension
    else:
      #q_uav = F.softmax(q_uav, dim=2)  # Probabilities with action over second dimension
      q_uuv = F.softmax(q_uuv, dim=2)  # Probabilities with action over second dimension
    return  q_uuv #q_uav,

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc_z' in name:
        module.reset_noise()
