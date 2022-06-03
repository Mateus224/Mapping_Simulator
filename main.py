import os
import torch
import argparse
import configparser
import render_UUV_env as env3D
import run_terminal as env_terminal
from Agents.lawnMower import lawnMower
from Agents.greedy import greedy

from Agents.random_agent import random_agent 
from Agents.ppo_agent import ppo_agent
from Env.env3d import  Env
from Env.load_env import Load_env


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--networkPath', default='network/', help='folder to put results of experiment in')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--load_net', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    # Environment
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--train', action='store_true', help='train the agent in the terminal')
    parser.add_argument('--episode_length', type=int, default=300, help='length of mapping environment episodes')
    
    # Visualization for c
    parser.add_argument('--gbp', action='store_false',
                        help='visualize what the network learned with Guided backpropagation')
    parser.add_argument('--gradCAM', action='store_false', help='visualize what the network learned with GradCAM')
    parser.add_argument('--gbp_GradCAM', action='store_false',
                        help='visualize what the network learned with Guided GradCAM')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize what the network learned with Guided GradCAM')
    parser.add_argument('--num_frames', type=int, default=80,
                        help='how many frames have to be stored in the prozessed video')
    # Agent
    parser.add_argument('--lawn_mower', action='store_true', help='comapre results to a lawn mower')
    parser.add_argument('--greedy', action='store_true', help='comapre results to a lawn mower')
    parser.add_argument('--random_agent', action='store_true', help='comapre results to a lawn mower')
    parser.add_argument('--ppo', action='store_true', help='on policy agent PPO')
    parser.add_argument('--dddqn', action='store_true', help='off policy agent DDDQN')
    parser.add_argument('--rainbow', action='store_true', help='off policy agent rainbow')
    parser.add_argument('--not_random_pose', action='store_false', help='comapre results to a lawn mower')
    parser.add_argument('--noisy-std', type=float, default=0.05, metavar='σ', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--priority_weight', type=float, default=0.2, metavar='beta', help='priority weight beta')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluation-interval', type=int, default=12000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--id', type=str, default='spectralNorm', help='Experiment ID')
    parser.add_argument('--checkpoint-interval', default=50000, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = True#args.enable_cudnn
    else:
        args.device = torch.device('cpu')
    return args


if __name__ == '__main__':
    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    args = parse()
    print(args.model_path)
    # make path
    results_dir = os.path.join('results', args.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    os.makedirs(args.networkPath, exist_ok=True)
    config = configparser.ConfigParser()
    config.read('config.ini')
    metrics = {'steps': [], 'rewards': [], 'entropy': []}
    #env_shape=[30,30,25] #changes has to be done in the memory.py manually 
    #uuv_shapes=[5,3]  #changes 
    #uav_shapes=[10,10]
    #sensor_range=6
    #print("cuda,",torch.cuda.is_available())

    #randompose=(config.getboolean("AUV", "random_pose"))
    env = Env(args, config)
    if args.lawn_mower:
        agent = lawnMower(args, env, metrics, results_dir)
    elif args.greedy:
        agent = greedy(args, env, metrics, results_dir)
    elif args.random_agent:
        agent = random_agent(args, env, metrics, results_dir)
    elif args.ppo:
        agent = ppo_agent(args, env, metrics, results_dir)
    elif args.dddqn:
        from Agents.DDDQN.DDDQN_agent import DDDQN_agent
        agent = DDDQN_agent(env, args, metrics, results_dir)
    elif args.rainbow:
        from Agents.multiagent_rainbow.agent import Multiagent_rainbow
        agent = Multiagent_rainbow(args, env)
    if args.do_render:
        print("-----------------------------rendering-----------------------------\n")
        env3D.init_render(args, agent, env, config)
        quit()
    print("-----------------------------not rendering-----------------------------\n")
    env_terminal.init(args, env, agent, config)

