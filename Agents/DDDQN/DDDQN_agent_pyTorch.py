from ..agent import Agent
import os
import math
import random
import numpy as np
#import tensorflow as tf
from collections import deque
#from tensorflow.keras.models import Sequential, Model, load_model
#from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Multiply, Lambda
#from tensorflow.keras.optimizers import RMSprop, Adam
#import tensorflow.keras.backend as K
#from tensorflow.keras.layers import Lambda, Concatenate
#from tensorflow.keras.utils import plot_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import operator
from tensorboardX import SummaryWriter


#from tensorflow.compat.v1.keras.backend import set_session

# random#.seed(1)
# np.random#.seed(1)
#tf.reset_default_graph()


# tf.set_random_seed(1)

# reference : https://github.com/tokb23/dqn/blob/master/dqn.py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
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

class QNetwork(nn.Module):
    def __init__(self, args, action_space):
        super(QNetwork, self).__init__()
        self.action_space = action_space

        self.convs = nn.Sequential(nn.Conv2d(5, 96, 6, stride=3, padding=0), nn.ReLU(), #nn.BatchNorm2d(96),nn.ReLU(),
                                    nn.Conv2d(96, 128, 3, stride=1, padding=0), nn.ReLU(), #nn.BatchNorm2d(128),nn.ReLU(),
                                    nn.Conv2d(128, 256, 2, stride=1, padding=0), nn.ReLU(), #nn.BatchNorm2d(256),nn.ReLU(), 
                                    nn.Flatten())
        num_features_before_fcnn = functools.reduce(operator.mul, list(self.convs(torch.rand(1, *(5,27,27))).shape))
        print(num_features_before_fcnn)

        self.dense = nn.Sequential(nn.Linear(num_features_before_fcnn, 512), nn.ReLU())
        
   
        self.fc_h_v = NoisyLinear(512, 512, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(512, 512, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(512, 1, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(512, action_space, std_init=args.noisy_std)

    def forward(self, x):
        map = self.convs(x)
        x = self.dense(map)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        q = v + a - a.mean(1, keepdim=True)  # Combine streams

        return q



    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


class DDDQN_agent(Agent):

    def __init__(self, env, args):
        import numpy as np
        from tensorflow.keras import backend as K
        import sys




        # parameters
        self.frame_x = env.xn
        self.frame_y = env.yn
        #self.frame_z = env.zn
        self.pose=env.pose
        self.num_steps = args.num_steps
        self.state_length = 1
        self.gamma = args.gamma
        self.exploration_steps = args.exploration_steps
        self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.initial_replay_size = args.initial_replay_size
        self.num_replay_memory = args.num_replay_memory
        self.batch_size = args.batch_size
        self.target_update_interval = args.target_update_interval
        self.train_interval = args.train_interval
        self.learning_rate = args.learning_rate
        self.min_grad = args.min_grad
        self.save_interval = args.save_interval
        self.no_op_steps = args.no_op_steps
        self.save_network_path = args.save_network_path
        self.save_summary_path = args.save_summary_path
        self.test_dqn_model_path = args.test_dqn_model_path
        self.exp_name = args.exp_name
        self.alpha=0.0
        self.re_alpha=0.0000005
        self.args=args
        self.device="cuda:0"
        


        #if args.optimizer.lower() == 'adam':
        #self.opt = Adam(lr=self.learning_rate)
        #else:
            #self.opt = RMSprop(lr=self.learning_rate, decay=0, rho=0.99, epsilon=self.min_grad)

        # environment setting

        self.env = env
        self.action_space = 12

        self.epsilon = self.initial_epsilon
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.exploration_steps
        self.t = 0

        # Input that is not used when fowarding for Q-value
        # or loss calculation on first output of model
        self.dummy_input = np.zeros((1, self.action_space))
        self.dummy_batch = np.zeros((self.batch_size, self.action_space))

        # for summary & checkpoint
        self.total_reward = 0.0
        self.total_q_max = 0.0
        self.total_loss = 0.0
        self.duration = 0
        self.episode = 0
        self.loss=None
        self.last_30_reward = deque()
        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)
        if not os.path.exists(self.save_summary_path):
            os.makedirs(self.save_summary_path)

        # Create replay memory
        self.replay_memory = Memory(self.num_replay_memory)

        # Create q network
        self.q_network = QNetwork(args,self.action_space).to(device)

        # Create target network
        self.target_network = QNetwork(args, self.action_space).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-4)

        # load model for testing, train a new one otherwise
        if args.test_dqn:
            self.target_network.load_state_dict(onlineQNetwork.state_dict())

        # Setup TensorBoard Writer
        import shutil

        string="./tensorboard/dddqn_/pytorch_tests"

        shutil.rmtree(string, ignore_errors=True)
        self.writer = SummaryWriter(string)



        #write_op = tf.summary.merge_all() 

    def init_game_setting(self):
        pass

    def train(self):
        self.entr=0
        self.terminal_count=0
        while self.t <= self.num_steps:
            self.terminal = False
            state,_,_ = self.env.reset()
            state= np.swapaxes(state,0,2)
            #state=torch.Tensor(state)
            #torch.FloatTensor(state).unsqueeze(0).to(device)
            while not self.terminal:
                last_state = state
                #state=torch.FloatTensor(state).unsqueeze(0).to(device)
                action = self.make_action(last_state, test=False)
                state, reward, self.terminal, _, self.entr = self.env.step(action)
                state= np.swapaxes(state,0,2)
                #state=torch.Tensor(state)
                #state=torch.FloatTensor(state).unsqueeze(0).to(device)
                self.run(last_state, action, reward, self.terminal, state)
                ## Losses


    def make_action(self, state, test=True):
        """
        ***Add random action to avoid the testing model stucks under certain situation***
        Return:
            action: int
                the predicted action from trained model
        """
        if not test:
            if self.epsilon >= random.random():# or self.t < self.initial_replay_size:
                action = random.randrange(self.action_space)
                
            else:
                action = np.argmax(self.q_network(state)) 
            # Anneal epsilon linearly over time
            if self.alpha>0.1:
                self.alpha = self.alpha - self.re_alpha
            if self.epsilon > self.final_epsilon and self.t >= self.initial_replay_size:
                if self.epsilon < 0.06:
                    self.epsilon-= self.epsilon_step/16
                elif self.epsilon < 0.12:
                    self.epsilon-= self.epsilon_step/8
                elif self.epsilon < 0.25:
                    self.epsilon-= self.epsilon_step/4
                elif self.epsilon < 0.5:
                    self.epsilon-= self.epsilon_step/2
                else:
                    self.epsilon -= self.epsilon_step
        else:
            if 0.01 >= random.random():
                action = random.randrange(self.action_space)
            else:
                #print(observation.shape)
                action = self.q_network.predict(state)
 
        return action

   
    def run(self, state, action, reward, terminal, observation):
        next_state = observation

        # Store transition in replay memory
        experience = state, action, reward, next_state, terminal
 
        self.replay_memory.store(experience)
        #if len(self.replay_memory) > self.num_replay_memory:
        #    self.replay_memory.popleft()

        if self.t >= self.initial_replay_size:
            # Train network
            if self.t % self.train_interval == 0:
                self.train_network()

            # Update target network
            if self.t % self.target_update_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.save_network_path + '/' + self.exp_name + '_' + str(self.t) + '.h5'
                self.q_network.save(save_path)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        #state=torch.FloatTensor(state).unsqueeze(0).to(device)
        #print(torch.max(self.q_network(state), dim=1, keepdim=True))
        #self.total_q_max += torch.max(self.q_network(state), dim=1, keepdim=True)
        self.duration += 1

        if terminal:
            self.terminal_count=self.terminal_count+1
            
            self.writer.add_scalar("Entropy", self.entr, global_step=self.terminal_count)

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

    def train_network(self):
        y_batch = []

        # Sample random minibatch of transition from replay memory
        tree_idx, minibatch, self.ISWeights_  = self.replay_memory.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(np.copy(minibatch[:]['state'])).to(device)
        action_batch = torch.FloatTensor(np.copy(minibatch[:]['action'])).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.copy(minibatch[:]['reward'])).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.copy(minibatch[:]['next_state'])).to(device)
        terminal_batch = torch.FloatTensor(np.copy(minibatch[:]['nonterminal'])).unsqueeze(1).to(device)
        
        with torch.no_grad():
            onlineQ_next = self.q_network(next_state_batch)
            targetQ_next = self.target_network(next_state_batch)
            online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma * targetQ_next.gather(1, online_max_action.long())
        print(self.q_network(state_batch).shape, action_batch.shape)
        loss = F.mse_loss(self.q_network(state_batch).gather(1, action_batch.long()), y_batch)
        self.optimizer.zero_grad()
        (torch.tensor(self.ISWeights_,device=self.device)*loss).mean().backward()
        self.optimizer.step()
        self.writer.add_scalar('loss', loss.item(), global_step=self.t)
        #writer.add_scalar('Norm. TD error', loss.item(), global_step=learn_steps)
        #with self.writer.as_default():
        #    tf.summary.scalar("Loss", self.loss[1], step=self.t)
        #    tf.summary.scalar("TD Error", np.mean(self.td_error), step=self.t)

        #self.total_loss += self.loss[1]
        loss=loss.detach().cpu().numpy()
        self.replay_memory.batch_update(tree_idx, loss)





Transition_dtype = np.dtype([('state', np.float32, (5,27,27)), ('action', np.int32), ('reward', np.float32),('next_state', np.float32, (5, 27,27)), ('nonterminal', np.bool_)])
blank_trans = (np.zeros((5, 27,27), dtype=np.float32), 0, 0.0, np.zeros((5, 27,27), dtype=np.float32),False)


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0
    
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.array([blank_trans] * self.capacity, dtype=Transition_dtype)  #np.zeros(capacity, dtype=object)
    
    
    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """
        
        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update the leaf
        self.update (tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
            
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node



class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # 0.6 Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.2 # 0.4  importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.0005
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)
        
    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p

        
    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []
        memory_c =np.empty((n,), dtype=Transition_dtype)
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        a=self.tree.tree[-self.tree.capacity:]
        p_min = np.min(a[np.nonzero(a)]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        #print(np.min(self.tree.tree[-self.tree.capacity:]),p_min)
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight             
            b_idx[i]= index
            
            #experience = [data]
           
            
            #memory_b.append(experience)
            memory_c[i]=data
        
        return b_idx, memory_c, b_ISWeights
    
    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        print(abs_errors, tree_idx)
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

