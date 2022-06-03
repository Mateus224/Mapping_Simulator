from ..agent import Agent
import os

import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Multiply, Lambda
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Concatenate, LSTM
from tensorflow.keras.utils import plot_model
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line

#from tensorflow.compat.v1.keras.backend import set_session

# random#.seed(1)
# np.random#.seed(1)
#tf.reset_default_graph()


# tf.set_random_seed(1)

# reference : https://github.com/tokb23/dqn/blob/master/dqn.py

class DDDQN_agent(Agent):

    def __init__(self, env, args, metrics, results_dir) :
        import numpy as np
        from tensorflow.keras import backend as K
        import sys




        # parameters
        self.frame_x = env.xn
        self.frame_y = env.yn
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
        self.ddqn = True# args.ddqn
        self.dueling = args.dueling
        self.alpha=0.0
        self.re_alpha=0.0000005
        self.metrics=metrics
        self.results_dir=results_dir


        if args.optimizer.lower() == 'adam':
            self.opt = Adam(lr=self.learning_rate)
        else:
            self.opt = RMSprop(lr=self.learning_rate, decay=0, rho=0.99, epsilon=self.min_grad)

        # environment setting

        self.env = env
        self.num_actions = 4

        self.epsilon = self.initial_epsilon
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.exploration_steps
        self.t = 0

        # Input that is not used when fowarding for Q-value
        # or loss calculation on first output of model
        self.dummy_input = np.zeros((1, self.num_actions))
        self.dummy_batch = np.zeros((self.batch_size, self.num_actions))

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
        if args.train:
            self.replay_memory = Memory(self.num_replay_memory)

        # Create q network
        self.q_network = self.build_network()

        # Create target network
        self.target_network = self.build_network()

        # load model for testing, train a new one otherwise
        if args.load_net:
            self.q_network.load_weights(self.test_dqn_model_path)
            self.log = open(self.save_summary_path + self.exp_name + '.log', 'w')

        else:
            self.log = open(self.save_summary_path + self.exp_name + '.log', 'w')

        # Set target_network weight
        self.target_network.set_weights(self.q_network.get_weights())

        # Setup TensorBoard Writer
        import shutil

        self.name=args.exp_name
        direc="./tensorboard/dddqn_/pose_v1"
        string=direc+self.name

        shutil.rmtree(string, ignore_errors=True)
        self.writer = tf.summary.create_file_writer(string)



        #write_op = tf.summary.merge_all() 

    def init_game_setting(self):
        pass

    def train(self):
        self.entr=505
        self.terminal_count=0
        while self.t <= self.num_steps:
            self.terminal = False
            observation,_,_ = self.env.reset()
            while not self.terminal:
                last_observation = observation
                action = self.make_action(last_observation, test=False)
                observation, reward, self.terminal, self.entr = self.env.step(action, agent="DDDQN")
                self.run(last_observation, action, reward, self.terminal, observation)
                ## Losses


    def make_action(self, observation, test=True):
        """
        ***Add random action to avoid the testing model stucks under certain situation***
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        if not test:
            self.rand=(3/self.entr)*(10*self.epsilon)
            ran=random.random()
            if self.rand<ran or 0.04<ran:
                action = np.argmax(self.q_network.predict([np.expand_dims(observation, axis=0),  self.dummy_input])[0])
            else:
                action = random.randrange(self.num_actions)
            #if self.epsilon >= random.random():# or self.t < self.initial_replay_size:
            #    action = random.randrange(self.num_actions)
                
            #else:
            #    action = np.argmax(self.q_network.predict([np.expand_dims(observation, axis=0),  self.dummy_input])[0])
            # Anneal epsilon linearly over time
            #if self.alpha>0.1:
            #    self.alpha = self.alpha - self.re_alpha
            if self.epsilon > self.final_epsilon and self.t >= self.initial_replay_size:
                if self.epsilon < 0.2:
                    self.epsilon-= self.epsilon_step/1024
                elif self.epsilon < 0.33:
                    self.epsilon-= self.epsilon_step/512
                elif self.epsilon < 0.5:
                    self.epsilon-= self.epsilon_step/256
                elif self.epsilon < 0.7:
                    self.epsilon-= self.epsilon_step/128
                elif self.epsilon < 1.:
                    self.epsilon-= self.epsilon_step/64
                elif self.epsilon < 1.3:
                    self.epsilon-= self.epsilon_step/32
                elif self.epsilon < 2.:
                    self.epsilon-= self.epsilon_step/16
                elif self.epsilon < 3.:
                    self.epsilon-= self.epsilon_step/8
                elif self.epsilon < 5.:
                    self.epsilon-= self.epsilon_step/4
                elif self.epsilon < 10.:
                    self.epsilon-= self.epsilon_step/2
                else:
                    self.epsilon -= self.epsilon_step
        else:
            #rand=random.random()
            #if self.t<60:
            #    self.t=self.t+1
            action = np.argmax(self.q_network.predict([np.expand_dims(observation, axis=0),  self.dummy_input])[0])
            #action = random.randrange(self.num_actions)
            #elif 0.1 >= rand:
            #    action = random.randrange(self.num_actions)
            #else:
                #print(observation.shape)
            #    action = np.argmax(self.q_network.predict([np.expand_dims(observation, axis=0),  self.dummy_input])[0])
 
        return action

    def build_network(self):


        input_frame = Input(shape=(self.frame_x, self.frame_y,4))

	
        action_one_hot = Input(shape=(self.num_actions,))
        conv1 = Conv2D(96, (6, 6), strides=(3, 3),  activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_frame)
        conv2 = Conv2D(128, (3, 3), strides=(1, 1),  activation=tf.keras.layers.LeakyReLU(alpha=0.01))(conv1)
        conv3 = Conv2D(256, (2, 2), strides=(1, 1),  activation=tf.keras.layers.LeakyReLU(alpha=0.01))(conv2)
        flat_feature = Flatten()(conv3)
        hidden_feature_comb=Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(flat_feature)

        if True:#self.dueling:
            value_hidden = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name = '2nd_v_dense')(hidden_feature_comb)
            #value_hidden = LSTM(512, name = 'LSTM_value_fc')(hidden_feature_comb)
            value = Dense(1, name = "value")(value_hidden)
            action_hidden = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name = '2nd_a_dense')(hidden_feature_comb)
            #action_hidden = LSTM(512, name = 'LSTM_action_fc')(hidden_feature_comb)
            action = Dense(self.num_actions, name = "action")(action_hidden)
            action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keepdims = True), name = 'action_mean')(action) 
            q_value_prediction = Lambda(lambda x: x[0] + x[1] - x[2], name = 'duel_output')([action, value, action_mean])
        select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
        #select_q_value_of_action = merge([q_value_prediction, action_one_hot], mode='mul',
        #                                 output_shape=(self.num_actions,))

        target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), output_shape=lambda_out_shape)(
            select_q_value_of_action)

        model = Model(inputs=[input_frame, action_one_hot], outputs=[q_value_prediction, target_q_value])

        # MSE loss on target_q_value only
        model.compile(loss=['mse', self.loos_function], loss_weights=[0.0, 1.0], optimizer=Adam(lr=0.0000001), run_eagerly=True)  # self.opt)

        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model

    def loos_function(self, target_Q, Q):
        loss= tf.reduce_mean(self.ISWeights_ * (tf.math.squared_difference(target_Q, Q)))
        self.td_error=np.abs((tf.cast((target_Q-Q)/Q, tf.float64)).numpy()) # np.abs((tf.cast((target_Q-Q)/ ( Q* self.entr), tf.float64)).numpy())
        return loss

   
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
                self.target_network.set_weights(self.q_network.get_weights())

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.save_network_path + '/' + self.exp_name + '_' + str(self.t) + '.h5'
                self.q_network.save(save_path)
                print('Successfully saved: ' + save_path)
        self.total_reward += reward
        self.total_q_max += np.max(self.q_network.predict([np.expand_dims(state, axis=0) ,self.dummy_input])[0])
        self.duration += 1

        if terminal:
            self.terminal_count=self.terminal_count+1
            with self.writer.as_default():
                tf.summary.scalar("Entropy", self.entr, step=self.terminal_count)
                tf.summary.scalar("Litter", self.env.litter, step=self.terminal_count)

                self.writer.flush()
            # Observe the mean of rewards on last 30 episode
            self.last_30_reward.append(self.total_reward)
            if len(self.last_30_reward) > 30:
                self.last_30_reward.popleft()

            # Log message
            if self.t < self.initial_replay_size:
                mode = self.name
            elif self.initial_replay_size <= self.t < self.initial_replay_size + self.exploration_steps:
                mode = self.name
            else:
                mode = self.name
            print(
                'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / RAND: {3:.5f} / REWARD: {4:2.3f} / ENTR: {5:2.4f} / EPS: {6:2.5f} / AVG_LOSS: {7:.5f} / MODE: {8} '.format(
                    self.episode + 1, self.t, self.duration, self.rand,
                    self.total_reward, self.entr, self.epsilon,
                    self.total_loss / (float(self.duration) / float(self.train_interval)), mode)) #, self.entr / ENTR: {8:.3f}
            print(
                'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / RAND: {3:.5f} / REWARD: {4:2.3f} / ENTR: {5:2.4f} / EPS: {6:2.5f} / AVG_LOSS: {7:.5f} / MODE: {8} '.format(
                    self.episode + 1, self.t, self.duration, self.rand,
                    self.total_reward, self.entr, self.epsilon,
                    self.total_loss / (float(self.duration) / float(self.train_interval)), mode), file=self.log)

            # Init for new game
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
        
        state_batch = np.copy(minibatch[:]['state'])
        action_batch = np.copy(minibatch[:]['action'])
        reward_batch = np.copy(minibatch[:]['reward'])
        next_state_batch = np.copy(minibatch[:]['next_state'])
        terminal_batch = np.copy(minibatch[:]['nonterminal'])


        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0
        # Q value from target network
        target_q_values_batch = self.target_network.predict([next_state_batch,self.dummy_batch])[0] #list2np(next_state_batch_position),list2np(next_state_batch_pose), self.dummy_batch])[0]

        # create Y batch depends on dqn or ddqn
        if True: #self.ddqn:
            next_action_batch = np.argmax(self.q_network.predict([next_state_batch, self.dummy_batch])[0], axis=-1)#list2np(next_state_batch_position),list2np(next_state_batch_pose), self.dummy_batch])[0],
                                          #axis=-1)
            for i in range(self.batch_size):
                y_batch.append(reward_batch[i] + (1 - terminal_batch[i]) * self.gamma * target_q_values_batch[i][
                    next_action_batch[i]])
            y_batch = list2np(y_batch)
        else:
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=-1)

        a_one_hot = np.zeros((self.batch_size, self.num_actions))
        for idx, ac in enumerate(action_batch):
            a_one_hot[idx, ac] = 1.0

        self.loss = self.q_network.train_on_batch([state_batch, a_one_hot], [self.dummy_batch, y_batch])
        with self.writer.as_default():
            tf.summary.scalar("Loss", self.loss[1], step=self.t)
            tf.summary.scalar("TD Error", np.mean(self.td_error), step=self.t)
            

        self.total_loss += self.loss[1]
        self.replay_memory.batch_update(tree_idx, self.td_error)


    def store_entropy(self,entropy, T):
        self.metrics['steps'].append(T)
        self.metrics['entropy'].append(entropy)
        self._plot_line(self.metrics['steps'], self.metrics['entropy'], 'Entropy', path=self.results_dir)

        


    def _plot_line(self, xs, ys_population, title, path=''):
        max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

        #ys = torch.tensor(ys_population, dtype=torch.float32)
        #ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
        #ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(x=xs, y=ys_population, line=Line(color=max_colour, dash='dash'), name='Max')
        #trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
        #trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
        #trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
        #trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

        plotly.offline.plot({
            'data': [trace_max],
            'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
        }, filename=os.path.join(path, title + '.html'), auto_open=False)


def list2np(in_list):
    return np.float32(np.array(in_list))


def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)







Transition_dtype = np.dtype([('state', np.float32, (27,27,4)), ('action', np.int32), ('reward', np.float32),('next_state', np.float32, (27,27,4)), ('nonterminal', np.bool_)])
blank_trans = (np.zeros((27,27, 4), dtype=np.float32), 0, 0.0, np.zeros((27,27,4), dtype=np.float32),False)


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
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

