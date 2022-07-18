import numpy as np
import time
from os.path import exists
from argparse import ArgumentParser
import torch  
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import functools
import operator
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from Agents.ppo.ppo_model import ResNet, ResBlock


"""
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 4
        self.conv1 = nn.Conv2d(4, 128, 5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 4, 27, 27)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 4, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(676, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(27*27*32, 4) #4=action.space
    
    def forward(self,s):
        x = F.relu(self.bn(self.conv(s))) # value head
        v = x.view(-1, 676)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.relu(self.fc2(v))

        #value=self.fc_z_v(F.relu(self.fc_h_v02(F.relu(self.fc_h_v01(x)))))
        #policy = F.softmax(self.fc_z_a(F.relu(self.fc_h_a02(F.relu(self.fc_h_a01(x))))), dim=-1)
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 5408)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class ResNetwork(nn.Module):
    def __init__(self, device):
        super(ResNetwork, self).__init__()
        self.device = device
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s=torch.Tensor(s).to(self.device)
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
"""
class Network(nn.Module):
    ''' Policy and value prediction from inner abstract state '''
    def __init__(self, device):
        super(Network, self).__init__()
        self.action_space = 8
        self.device = device

        self.convs = nn.Sequential(nn.Conv2d(4, 128, 5, stride=2, padding=2), nn.ReLU(), #nn.BatchNorm2d(96),nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(), #nn.BatchNorm2d(128),nn.ReLU(),
                                    nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(), #nn.BatchNorm2d(128),nn.ReLU(),
                                    nn.Conv2d(256, 512, 2, stride=1, padding=1), nn.ReLU(), #nn.BatchNorm2d(256),nn.ReLU(), 
                                    nn.Flatten())
        num_features_before_fcnn = functools.reduce(operator.mul, list(self.convs(torch.rand(1, *(4,38,38))).shape))
        print(num_features_before_fcnn)
        self.dense = nn.Sequential(nn.Linear(num_features_before_fcnn, 512), nn.ReLU())
        #self.fc_h_v = NoisyLinear(512, 512, std_init=args.noisy_std)
        #self.fc_h_a = NoisyLinear(512, 512, std_init=args.noisy_std)
        #self.fc_z_v = NoisyLinear(512, 1, std_init=args.noisy_std)
        #self.fc_z_a = NoisyLinear(512, action_space, std_init=args.noisy_std)
        self.fc_h_v01 = nn.Linear(512, 512)
        self.fc_h_v02 = nn.Linear(512, 512)
        self.fc_h_v03 = nn.Linear(512, 512)
        self.fc_h_v04 = nn.Linear(512, 512)

        self.fc_h_a01 = nn.Linear(512, 512)
        self.fc_h_a02 = nn.Linear(512, 512)
        self.fc_h_a03 = nn.Linear(512, 512)
        self.fc_h_a04 = nn.Linear(512, 512)

        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_a = nn.Linear(512, self.action_space)


    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        map = self.convs(x)
        x = self.dense(map)

        value=self.fc_z_v(F.relu(self.fc_h_v03(F.relu(self.fc_h_v02(F.relu(self.fc_h_v01(x)))))))
        policy = F.softmax(self.fc_z_a(F.relu(self.fc_h_a03(F.relu(self.fc_h_a02(F.relu(self.fc_h_a01(x))))))), dim=-1)

        #value = self.fc_z_v(F.relu(self.fc_h_v(x))) 
        #policy = F.softmax(self.fc_z_a(F.relu(self.fc_h_a(x))), dim=-1)
        

        #p = F.relu(self.p1(x))
        #p  = F.relu(self.p2(p))
        #policy = F.softmax(self.actor(p), dim=-1)
        
        #v  = F.relu(self.v1(x))
        #v  = F.relu(self.v2(v))
        #value = self.critic(v)
        
        return policy, value


class Memory():
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ppo_agent():

    def __init__(self, args, env, metrics, results_dir):
        self.metrics=metrics
        self.results_dir=results_dir
        self.writer = SummaryWriter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        self.env = env
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #parser = ArgumentParser(description='Run an algorithm on the environment')
        #parser.add_argument('--train', dest='train', action='store_true',
                            #help='Train our model.')



        #args = parser.parse_args()
        self.model_path = args.test_dqn_model_path
        
        self.network = ResNet(self.device, 4, ResBlock, [2,2,2,2], useBottleneck=False).to(self.device) # Network(self.device).to(self.device) #


        if exists(self.model_path) == True:
            checkpoint = torch.load(self.model_path)
            self.network.load_state_dict(checkpoint)  
            print("Loaded Network")
            #training_steps =  checkpoint['training_step']   

        if args.train:
            self.train()
        
        else:
            self.play()




    

    def get_action(self, state, action_probs, memory):
            dist = Categorical(action_probs)
            action = dist.sample()
            #state = torch.from_numpy(state)     
            state =  torch.squeeze(state)
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

            return action.item()

    def update(self, memory, next_state, gamma, epochs, eps_clip, loss_steps, optimizer, network):   
            # TD estimate of state rewards:
            rewards = []
            _, value = network.forward(next_state)
            discounted_reward = value.detach()
            
            for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            
            # Normalizing the rewards:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            
            # convert list to tensor
            old_states = torch.stack(memory.states)
            old_actions = torch.stack(memory.actions).to(self.device).detach()
            old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
            
            # Optimize policy for K epochs:
            for epoch in range(epochs):
                # Evaluating old actions and values :
                policy, value = network.forward(old_states)
                value = torch.squeeze(value)
                dist = Categorical(policy)
                logprobs = dist.log_prob(old_actions)
                dist_entropy = dist.entropy()

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs)
                    
                # Finding Surrogate Loss:
                advantages = rewards - value.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

                mse = nn.MSELoss()
                actor_loss = torch.min(surr1, surr2)
                critic_loss = 0.5 * mse(value, rewards)
                entropy = 0.01 * dist_entropy
                loss = -actor_loss + critic_loss - entropy
                        
                # take gradient step
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()  
                
                
                # Log the loss
            self.writer.add_scalar("1. Total Loss", loss.mean(), loss_steps)
            self.writer.add_scalar("2. Policy Loss", actor_loss.mean(), loss_steps)
            self.writer.add_scalar("3. Value Loss", critic_loss.mean(), loss_steps)

    def train(self):
        
        max_episodes = 10000
        gamma = 0.99
        learning_rate = 1e-7
        #betas = (0.9, 0.999)
        eps_clip = 0.2
        epochs = 5
        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)#, betas=betas)
        
        # logging variables
        max_timesteps = 800
        update_timestep = 5      # update policy every n timesteps
        log_interval = 20
        running_reward = 0
        avg_length, timestep = 0, 0

        mem = Memory()
        start = time.time()
        loss_steps =0
        
        
        for episode in range(1, max_episodes + 1):
            state,_,_ = self.env.reset()
            state=torch.Tensor(state)
            state= np.swapaxes(state,0,2)
            state = state.unsqueeze(0)
            
            #print(state.shape)
            

            for t in range(max_timesteps):

            
                policy, _ = self.network.forward(state)
                action = self.get_action(state, policy, mem)
                
                state, reward, done, _ = self.env.step(action, agent="PPO")
                #print(reward, done, self.env.t)
                state=torch.Tensor(state)
                state= np.swapaxes(state,0,2)
                state = state.unsqueeze(0)                
                
                # Saving reward and is_terminal:
                mem.rewards.append(reward)
                mem.is_terminals.append(done) 
                
                timestep +=1
                # update if its time
                if timestep % update_timestep == 0:
                    loss_steps += 1
                    self.update(mem, state, gamma, epochs, eps_clip, loss_steps, optimizer, self.network)
                    mem.clear_memory()
                    self.writer.flush()
            
                running_reward += reward
                if done:
                    break    
            
            avg_length += t

            if episode % 40== 0:
                #Save Model
                torch.save(self.network.state_dict(), self.model_path)   

            if episode % log_interval == 0:
                avg_length = int(avg_length/log_interval)
                running_reward = running_reward/log_interval

                print("{}. Time: {}, Avg Length: {}, Avg Rewards: {}".format(episode, int(time.time()-start) , avg_length, running_reward))

                running_reward = 0
                avg_length = 0
                start = time.time()
                self.writer.flush()
        
                        

    def play(self):

        with torch.no_grad():
        
            play_ep = 20
            max_timesteps = 300
            try:
                avg = 0
                for e in range(play_ep):
                    state,_,_ = self.env.reset()
                    reward_sum = 0.0

                    for _ in range(max_timesteps):
                        #env.render(mode='rgb_array')
                        state=torch.Tensor(state)
                        state= np.swapaxes(state,0,2)
                        state = state.unsqueeze(0)
                        policy, _ = self.network.forward(state)
                        dist = policy.cpu().detach().numpy() 
                        action = np.argmax(dist)

                        state, reward, done, _ = self.env.step(action, agent="PPO")
                        reward_sum += reward
                        if done:
                            break
                    
                    avg += reward_sum   

                    print("{}. Reward: {}".format(e+1, int(reward_sum)))

                print(f"NGB has played {play_ep} Episodes and the average Reward is {int(avg/play_ep)}.") 

            except KeyboardInterrupt:
                    print("Received Keyboard Interrupt. Shutting down.")

                


    #env = create_env()
    #inp = env.observation_space.shape[0]
    #output = env.action_space.n












