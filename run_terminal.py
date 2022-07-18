from Env.load_env import Load_env
from Env.env3d import  Env
#from Agents.DDDQN.DDDQN_agent_pyTorch import DDDQN_agent
from Agents.DDDQN.DDDQN_agent import DDDQN_agent
from Agents.multiagent_rainbow.test import test
from Agents.multiagent_rainbow.agent import Multiagent_rainbow
from Agents.multiagent_rainbow.memory import ReplayMemory
from Agents.lawnMower.lawnMower import lawnMower
from Agents.greedy.greedy import greedy
from random import random
import torch
import os
import time
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter

def init(args, env, agent, config):

    global obs


    List2_columns=1
    List1_row=40
    s = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    l = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    e = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    metrics = {'steps': s, 'litter': l, 'entropy': e}#, 'litter_amount':res}
    t=np.arange(800)
    run_noCrash=False
    episode=0
    all_actions=False
    if args.train:
        agent.train()
        if type(agent)==Multiagent_rainbow:
            string="./tensorboard/rainbow/paper_smallNN"
            writer = SummaryWriter(string)
            timeout=False
            mem = ReplayMemory(args, args.num_replay_memory)
            priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
            results_dir = os.path.join('results', args.id)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            T, done = 0, False
            sum_reward=0
            state, _ = env.reset(h_level=False)
            for T in trange(1, int(args.num_steps)):
                if done or timeout:

                    print(sum_reward,'  ------  Litter:', np.sum(env.reward_map_bel))#,(sum_reward+(0.35*done))/(env.start_entr_map))
                    #print(np.nanmax(env.loaded_env.map_2_5D),'max height')
                    sum_reward=0
                    state, _ = env.reset(h_level=False)


                    litter=env.litter
                    print(sum_reward,'------', litter)#,(sum_reward+(0.35*done))/(env.start_entr_map))
                    #print(np.nanmax(env.loaded_env.map_2_5D),'max height')
                    sum_reward=0
                    state, _ = env.reset(h_level=False)
                    episode+=1
                    writer.add_scalar("Litter", litter, episode)

                #action = agent.epsilon_greedy(T,2, state)

                action = agent.epsilon_greedy(T,3000000, state, all_actions)

                #if T % args.replay_frequency == 0:

                agent.reset_noise()  # Draw a new set of noisy weights

                #agent.reset_noise()  # Draw a new set of noisy weights

                  # Choose an action greedily (with noisy weights)

                next_state, reward, done, actions, sim_i, timeout = env.step(action,all_actions=all_actions, h_level=False, agent="rainbow")  # Step
                sum_reward=sum_reward+reward
                 # Append transition to memory

                # Train and test
                if sim_i>0:
                    for j in range(sim_i-1):
                        mem.append(state, actions[j], 0, True)
                mem.append(state, actions[sim_i], reward, done) 
                if T >= 100000:#args.learn_start:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                #if T % args.replay_frequency == 0:
                    agent.learn(mem,T,writer)  # Train with n-step distributional double-Q learning

                    #if T % args.evaluation_interval == 0:
                    #    agent.eval()  # Set DQN (online network) to evaluation mode
                    #    avg_reward, avg_Q = test(args, T, agent, val_mem, metrics, results_dir)  # Test
                    #    log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                    #    agent.train()  # Set DQN (online network) back to training mode

                        # If memory path provided, save it
                    #    if args.memory is not None:
                    #        save_memory(mem, args.memory, args.disable_bzip_memory)

                    # Update target network
                    if T % args.target_update == 0:
                        agent.update_target_net()

                    # Checkpoint the network
                    if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                        agent.save(results_dir, 'checkpoint.pth')

                state = next_state
    else :  
        f_p=0 
        experiments=0
        simulate=False
        agent.eval()
        for i in range(List1_row):
            j=0
            done=False
            state, _ = env.reset(h_level=False, validation=True)
            if type(agent)!=DDDQN_agent:
                simulate=True
            else:
                run_noCrash=True
                #pass
            
            #speed test
            t0 = time.time()
           
            while not done:
                j=j+1
                
                action = agent.make_action(state, all_actions)
                
                next_state, reward, done, actions, sim_i, timeout = env.step(action,all_actions, h_level=False, agent="rainbow")
                #if(j%1000==0):
                t1 = time.time()
                #print(t1-t0, 'tot')

                if j<=349:# and False== done:
                    metrics['steps'][i].append(env.t)

                    #print(env.t)
                    #metrics['entropy'][i].append(entropy)
                    metrics['litter'][i].append(env.litter)
                    if done:
                        print("Error")

                #    belief=env.belief
                #    reward+= reward
                else:
                    false_positive=np.where(env.reward_map_bel-env.loaded_env.map_2_5D[:,:,1]>0,1,0)
                    
                    f_p=np.sum(false_positive)+f_p
                    print(f_p)
                    print(env.litter)
                    #metrics['litter'][i].append(env.litter)
                    #metrics['steps'][i].append(j)
                    done=True
            #experiments= experiments+1
        #print(metrics['entropy'])
        #_plot_line(t, metrics['entropy'], 'Entropy')
        _plot_line(t, metrics['litter'], 'Litter')
                    

def _plot_line2(xs, ys_population, title, path='/home/mateus/'):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'
    max_step=np.zeros(len(ys_population[0]))
    min_step=np.zeros(len(ys_population[0]))
    std_step=np.zeros(len(ys_population[0]))
    mean_step=np.zeros(len(ys_population[0]))
    ys_population=np.array(ys_population)
    for _steps in range(len(ys_population[0])):
        episode=ys_population[:,_steps]
        #print(episode)
        max_step[_steps]=np.max(episode)
        min_step[_steps]=np.min(episode)
        std_step[_steps]=np.std(episode)
        mean_step[_steps]=np.mean(episode)

    upper_step, lower_step = mean_step + std_step, mean_step - std_step

    trace_max = Scatter(x=xs, y=max_step, line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=upper_step, fillcolor=std_colour, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=mean_step,  fill='tonexty',  fillcolor=std_colour,line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=lower_step, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=min_step, line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_max,trace_upper, trace_mean, trace_lower, trace_min],
        'layout': dict(font_size=50,title=title, xaxis={'title': 'Step'}, yaxis={'title': title}),
    }, filename=os.path.join(path, title + '.html'), auto_open=False)


def _plot_line(xs, ys_population, title, path='/home/mateus/'):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)

    ys_ = ys[0].squeeze()

    ys_min, ys_max, ys_mean, ys_std = np.amax(ys_population, axis=0), np.amin(ys_population, axis=0), ys.mean(0).squeeze(), ys.std(0).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std


    trace_max = Scatter(x=xs, y=ys_max, fillcolor=std_colour,  line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), fillcolor=std_colour, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False) #line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean,  fill='tonexty',  fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(),fill='tonexty',  fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_max,trace_upper, trace_mean, trace_lower, trace_min],
        'layout': dict(font_size=40,title=title, xaxis={'title': 'Step'}, yaxis={'title': title}),
    }, filename=os.path.join(path, title + '.html'), auto_open=False)