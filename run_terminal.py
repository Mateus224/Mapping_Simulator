from Env.load_env import Load_env
from Env.env3d import  Env
#from Agents.DDDQN.DDDQN_agent_pyTorch import DDDQN_agent
from Agents.DDDQN.DDDQN_agent import DDDQN_agent
from Agents.multiagent_rainbow.test import test
from Agents.multiagent_rainbow.agent import Multiagent_rainbow
from Agents.multiagent_rainbow.memory import ReplayMemory
from Agents.lawnMower import lawnMower
from Agents.greedy import greedy
from random import random
import torch
import os
import time
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import numpy as np
from tqdm import trange

def init(args, env, agent, config):

    global obs
    mem = ReplayMemory(args, 5000000)
    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
    List2_columns=1
    List1_row=40
    s = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    l = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    e = [[ 0 for x in range(List2_columns)] for i in range (List1_row)]
    metrics = {'steps': s, 'litter': l, 'entropy': e}#, 'litter_amount':res}
    t=np.arange(800)
    run_noCrash=False
    if args.train:
        agent.train()
        if type(agent)==Multiagent_rainbow:
            results_dir = os.path.join('results', args.id)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            T, done = 0, True
            sum_reward=0
            for T in trange(1, int(args.num_steps)):
                if done:
                    print(sum_reward,'------',sum_reward/env.start_entr_map)
                    sum_reward=0
                    state, _ = env.reset()

                #if T % args.replay_frequency == 0:
                #agent.reset_noise()  # Draw a new set of noisy weights

                  # Choose an action greedily (with noisy weights)
                if T<80000:
                    action=np.random.randint(8)
                else:
                    action = agent.epsilon_greedy(T,state)
                next_state, reward, done, _ = env.step(action)  # Step
                sum_reward=sum_reward+reward
                mem.append(state, action, reward, done)  # Append transition to memory

                # Train and test
                if T >= 60000:#args.learn_start:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                #if T % args.replay_frequency == 0:
                    agent.learn(mem)  # Train with n-step distributional double-Q learning

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
        experiments=0
        simulate=False
        for i in range(List1_row):
            j=0
            done=False
            obs, _ = env.reset(validation=True)
            if type(agent)!=DDDQN_agent:
                simulate=True
            else:
                run_noCrash=True
                #pass
            
            #speed test
            t0 = time.time()
            while not done:
                j=j+1
                
                action = agent.make_action(obs)
                
                obs, reward, donea, entropy = env.step(action, lm=False, agent="DDDQN")
                #if(j%1000==0):
                t1 = time.time()
                print(t1-t0, 'tot')

                if j<=241:# and False== done:
                    metrics['steps'][i].append(env.t)
                    metrics['entropy'][i].append(entropy)
                    metrics['litter'][i].append(env.litter)

                #    belief=env.belief
                #    reward+= reward
                else:
                    print(i)
                    print(env.litter)
                    #metrics['litter'][i].append(env.litter)
                    #metrics['steps'][i].append(j)
                    done=True
            #experiments= experiments+1
        #print(metrics['entropy'])
        _plot_line(t, metrics['entropy'], 'Entropy')
        _plot_line(t, metrics['litter'], 'Litter')
                    

def _plot_line(xs, ys_population, title, path='/home/mateus/'):
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