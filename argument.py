def add_arguments(parser):

    parser.add_argument('--frame_width', type=int, default = 84, help='Resized frame width')
    parser.add_argument('--frame_height', type=int, default = 84, help='Resized frame height')
    parser.add_argument('--num_steps', type=int, default = 50e6, help='Number of episodes the agent plays')
    parser.add_argument('--state_length', type=int, default = 4, help='Number of most recent frames to produce the input to the network')
    parser.add_argument('--gamma', type=float, default = 0.99, help='Discount factor')
    parser.add_argument('--exploration_steps', type=int, default =50000, help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')#100000
    parser.add_argument('--initial_epsilon', type=float, default = 1.99, help='Initial value of epsilon in epsilon-greedy')
    parser.add_argument('--final_epsilon', type=float, default = 0.1, help='Final value of epsilon in epsilon-greedy')
    parser.add_argument('--initial_replay_size', type=int, default =100000, help='Number of steps to populate the replay memory before training starts')

    parser.add_argument('--num_replay_memory', type=int, default = 1500000, help='Number of replay memory the agent uses for training')

    parser.add_argument('--batch_size', type=int, default = 32, help='Mini batch size')
    parser.add_argument('--target_update_interval', type=int, default = 10000, help='The frequency with which the target network is updated')
    parser.add_argument('--train_interval', type=int, default = 4, help='The agent selects 4 actions between successive updates')
    parser.add_argument('--learning_rate', type=float, default = 0.000001, help='Learning rate used by RMSProp')
    parser.add_argument('--min_grad', type=float, default = 1e-8, help='Constant added to the squared gradient in the denominator of the RMSProp update')
    parser.add_argument('--save_interval', type=int, default = 25000, help='The frequency with which the network is saved')
    parser.add_argument('--no_op_steps', type=int, default = 10, help='Maximum number of "do nothing" actions to be performed by the agent at the start of an episode')
    parser.add_argument('--save_network_path', type=str, default = "saved_dqn_networks/", help='')
    parser.add_argument('--save_summary_path', type=str, default = "dqn_summary/", help='')

    parser.add_argument('--model_path', type=str, default = "results/twoActions1_100/checkpoint.pth", help='model used during testing / visulization') #testmoreFilters.h5
    parser.add_argument('--exp_name', type=str, default = "", help='')
    parser.add_argument('--gpu_frac', type=float, default = 1.0, help='Set GPU use limit for tensorflow')
    parser.add_argument('--ddqn', type=bool, default = False, help='Set True to apply Double Q-learning')
    parser.add_argument('--dueling', type=bool, default = False, help='Set True to apply Duelinng Network')
    parser.add_argument('--optimizer',type=str, default='adam', help='Optimizer (Adam or Rmsp)')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
    ######################     ssh -L 6006:localhost:6006  zsofia@10.20.7.59
    #Make video arguments#     http://localhost:6006/#scalars
    ######################     python3.7 -m tensorboard.main --logdir tensorboard/dddqn/1
    #python3 main.py --test_dqn --test_dqn_model_path saved_dqn_networks/new_env_125000.h5 --do_render
    parser.add_argument('-f', '--num_frames', default=100, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=0, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='./movies/', type=str, help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')

    return parser
