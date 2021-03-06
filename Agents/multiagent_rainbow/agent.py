# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torchsummary import summary
from torch.nn.utils import clip_grad_norm_
from torchsummary import summary
from numpy.random import default_rng
rng = default_rng(12345)
#
from Agents.multiagent_rainbow.model import DQN, DQN_ResNet, ResBlock
from tensorboardX import SummaryWriter
from numpy.random import default_rng
rng = default_rng(12345)


class Multiagent_rainbow():
  def __init__(self, args, env):

    self.action_space = 8#env.action_space()
    self.atoms = args.atoms
    self.Vmin = -0.35#args.V_min
    self.Vmax = 72#args.V_max

    self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (self.Vmax - self.Vmin) / (self.atoms - 1)
    self.batch_size = 32 #args.batch_size
    self.n = 1#args.multi_step
    self.discount = 0.99 #args.discount
    self.norm_clip = args.norm_clip
    self.device=args.device
    self.channels=5

    #self.online_net = DQN(args, self.action_space).to(device=args.device)
    self.online_net = DQN_ResNet(args, self.action_space, ResBlock, [2,2,2,2], self.channels).to(self.device)
    summary(self.online_net, (5, 32, 32))
    if args.load_net:  # Load pretrained model if provided
      if os.path.isfile(args.model_path):
        print(args.model_path)
        checkpoint = torch.load(args.model_path, map_location=torch.device(self.device))
        self.online_net.load_state_dict(checkpoint, strict=True)  
        #state_dict = torch.load(args.model_path, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        #if 'conv1.weight' in state_dict.keys():
        #  for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
        #    state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
        #    del state_dict[old_key]  # Delete old keys for strict load_state_dict
        #self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model ")
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model_path)

    self.online_net.train()

    #self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.target_net = DQN_ResNet(args,self.action_space, ResBlock, [2,2,2,2], self.channels).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def make_action(self, state, all_actions=False):
    state= np.swapaxes(state,0,2)
    self.eval()
    state=torch.Tensor(state).to(device=self.device)
    with torch.no_grad():
      if all_actions:
        return torch.squeeze((self.online_net(state.unsqueeze(0)) * self.support).sum(2)).cpu().detach().numpy()
      else:
        return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
    
  def get_actions(self, state):
    state= np.swapaxes(state,0,2)
    self.eval()
    state=torch.Tensor(state).to(device=self.device)
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2)

  # Acts with an ??-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ?? can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem, T, writer):
    # Sample transitions
    self.train()
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
    

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ??; ??online)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; ??online)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ??; ??online)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ??; ??online))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; ??online))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ??; ??target)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; ??online))]; ??target)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (??^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / ??z
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()


    writer.add_scalar("Loss", loss.mean().detach().cpu().numpy(), T)



    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()


  def epsilon_greedy(self,T, max,state, all_actions=False):
    if (max>T):
      prob=(max-(T))/(max)



      if rng.random()<prob:
        if all_actions:
          action=torch.rand(self.action_space)
        else:
          action=np.random.randint(8)
      else:
        action=self.make_action(state, all_actions)
    else:
      action=self.make_action(state, all_actions)
    return action
  
