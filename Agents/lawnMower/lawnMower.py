import Env.pytransform3d_.visualizer as pv
from Env.pytransform3d_.rotations import *
from Env.pytransform3d_.transformations import *
from Env.pytransform3d_.batch_rotations import *
import open3d as o3d
from Env.pytransform3d_.plot_utils import plot_box
import numpy as np
from Env.load_env import Load_env
from Env.env3d import  Env
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import os
import time


class lawnMower():


    def __init__(self, args, env, metrics, results_dir, randompose=False, num_beams=15):
        self.env=env
        self.direction=0
        self.counter=0
        self.prev_dircetion=0
        self.metrics=metrics
        self.results_dir=results_dir
        self._2Dright=0
        self._2Dup=2
        self._2Dleft=1
        self._3Dup=4
        self._3Ddown=5
        self.action=self._2Dright
        self.hight=6
        self.env.reset(h_level=False)
        self.done=False
        self.tmp=False
        self._2DEnv=True
    
    def reset(self):
        self.direction=0
        self.counter=0
        self.prev_dircetion=0
        self._2Dright=0
        self._2Dup=2
        self._2Dleft=1
        self._3Dup=4
        self._3Ddown=5
        self.action=self._2Dright
        self.hight=6
        self.done=False
        self.tmp=False
        self._2DEnv=True
        self.counter=0



    def step_in_direction(self,hight_o_ground):
        if not self.tmp:
            self.done = not self.env.agentDispatcher.sim_legal_action(self.action)
        if self._2DEnv:  
            self.lawnMowerTraject()
        else:
            if (hight_o_ground==self.hight):
                self.lawnMowerTraject()  
            elif (hight_o_ground)<self.hight:
                self.action= self._3Dup
                if self.done :
                    self.tmp=True
            elif (hight_o_ground)>self.hight:
                self.action= self._3Ddown
                if self.done:
                    self.tmp=True
        return self.action

    def lawnMowerTraject(self):
        if self.done==True and self.direction==0:
                self.direction=1
                self.action=self._2Dup
                self.env.step(self.action,h_level=False)
                self.env.step(self.action,h_level=False)
                self.tmp=False
        elif self.done==False and self.direction==0:
            self.action=self._2Dright
            
        elif self.done==True and self.direction==1:
            self.action=self._2Dup
            self.env.step(self.action,h_level=False)
            self.env.step(self.action,h_level=False)
            #elf.env.step(self.action)
            self.direction=0
            self.tmp=False
        elif self.done==False and self.direction==1:
            self.action=self._2Dleft

    def make_action(self, obs,_):
        self.counter=self.counter+1
        attitude=self.env.agentDispatcher.uav.pose.pose_matrix[:3,3]
        #print(attitude)
        a_x=int(np.rint(attitude[0]))
        a_y=int(np.rint(attitude[1]))
        a_z=int(np.rint(attitude[2]))
        map=self.env.loaded_env.map_2_5D[:,:,0]
        ground_UUV = map[a_x,a_y]
        hight_o_ground= a_z-ground_UUV
        action = self.step_in_direction(hight_o_ground)
        return action


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