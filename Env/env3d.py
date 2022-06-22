from pickle import NONE
import numpy as np
from Env.load_env import Load_env
from Env.agent import AgentDispatcher

from collections import deque
import cv2
import torch
from scipy.special import logsumexp
import warnings
import traceback , gc
import json
import Env.agent
import time


def safe_log(x):
    if x <= 0.:
        return 0.
    return np.log(x)
safe_log = np.vectorize(safe_log)

class Env(object):
    def __init__(self, args, config):
        json_env_shape=config.get('ENV','shape')
        list_env_shape=json.loads(json_env_shape)
        self.env_shape=np.array(list_env_shape)

        self.xn = self.env_shape[0]
        self.yn = self.env_shape[1]
        self.zn = self.env_shape[2]
        self.map=np.zeros((self.xn,self.yn,self.zn,3))
        self.sub_env=config.getboolean('ENV','sub_env')
        if self.sub_env:
            self.sub_env_size=config.getint('ENV','sub_env_size')
            assert (self.xn%self.sub_env_size and self.yn%self.sub_env_size)==0 , 'The sub environment must be a multiple of the original environment. Check your config.ini file.'
            self.sub_bel=np.zeros((self.sub_env_size,self.sub_env_size))
            self.sub_ent=np.zeros((self.sub_env_size,self.sub_env_size))
            self.sub_uav_pos=np.zeros((3))
        self.viewer = None
        self.ent = None
        self.episode=0
        self.agentDispatcher=AgentDispatcher(config, self.xn, self.yn, self.zn)
        self.start_entr_map=167
        self.uav_state_pos=np.zeros(7)
        self.uuv_state_pos=np.zeros(7)
        



    def reset(self, h_level=True, episode_length=750, validation=False):
        self.entr_map=self.calc_entropy(np.ones((self.xn, self.yn))/2)
        self.timeout=False
        gc.collect()# garbege collector
        self.done=False
        self.episode_length=episode_length
        if not validation:
            map_int=np.random.random_integers(111)
            upSideDown=True
            if map_int==2 or map_int==3 :
                upSideDown=False
            map=str(map_int)
            format='.xyz'
            folder='Env/xyz_env/xyz/' 
        else:
            self.episode=self.episode+1
            map=str(self.episode)
            format='_val.xyz'
            folder='Env/xyz_env/xyz/val/'
            upSideDown=True
        pc=folder+map+format
        self.loaded_env=Load_env(self.env_shape)
        self.map, self.hashmap = self.loaded_env.load_VM(pc=pc, upSideDown=upSideDown)
        self.belief=np.ones((self.xn,self.yn))/2
        if self.sub_env:
            self.sub_belief=np.ones((int(self.xn/self.sub_env_size), int(self.yn/self.sub_env_size)))/2
        self.tmp_coordinate_storage=np.zeros((self.env_shape[0],self.env_shape[1], self.env_shape[2]))
        self.reward_map_entr=np.zeros((self.env_shape[0],self.env_shape[1]))
        self.reward_map_bel=np.zeros((self.env_shape[0],self.env_shape[1]))

        self.agentDispatcher.reset(self.loaded_env)
        self.position2_5D=np.zeros((self.xn, self.yn))-2
        self.position2_5D_R=np.zeros((self.xn, self.yn))
        self.uuv_last_poseX,self.uuv_last_poseY= 0,0
        self.uav_last_poseX,self.uav_last_poseY= 0,0
        self.litter=0
        self.litter_amount=0

        self.entr_old = self.calc_entropy(np.ones((self.xn, self.yn))/2)
        self.t = 0
        self.litter=0
        if h_level:
            plane=np.random.randint(10,12)
            self.obstacles=np.where(self.loaded_env.map_2_5D[:,:,0]>=plane, 1, 0)
            
            self.agentDispatcher.uav.pose.pose_matrix[2,3]=plane
            obst=np.random.randint(15, size=6)
            self.obstacles[obst[0],obst[1]]=1
            self.obstacles[obst[0]+1,obst[1]]=1
            self.obstacles[obst[0],obst[1]+1]=1
            self.obstacles[obst[2],obst[3]]=1
            self.obstacles[obst[2]+1,obst[3]]=1
            self.obstacles[obst[2]+1,obst[3]+1]=1
            self.obstacles[obst[4],:]=0
            self.obstacles[obst[4]+1,:]=0
            self.obstacles[:,obst[5]]=0
            self.obstacles[:,obst[5]+1]=0
            self.entr_map=np.where(self.obstacles==1, 0, self.entr_map)
            
            self.loaded_env.map_2_5D[:,:,0]=np.where(self.obstacles==1, plane, 0)
            self.start_entr_map = np.sum(self.entr_map)
            #print(self.loaded_env.map_2_5D[:,:,0])
            return self.get_hLevel_observation() , self.map
        return self.get_observation() , self.map


    def renderMatrix(self, matrix, name="image"):
        matrix=matrix*255   
        matrix= matrix.astype(np.uint8)
        cv2.imshow(name,matrix)
        cv2.waitKey(1)

    def calc_entropy(self, b_t):
        entropy = - (b_t * safe_log(b_t) + (1 - b_t) * safe_log(1 - b_t))

        return entropy

    def calc_sum_entropy(self, obs):
        np.sum(obs)
        return

    def observation_size(self):
        return 2 * self.N - 1



    def get_observation(self, single_agent=True):
        """This method is for the use of a multiagent (2 agents) DRL algorithm 
        which returns the state of the agent, which contains the posisition of 
        the angent labaled with in a matrix initialised by -2. To differentiate
        between both agents (which is not needed in our case because they are 
        alway on different heights), however we are adding to the uav +2.
        Beside the state we have a 2.5D representation of the map and the belief
        
        TODO: Put the agents into a list. 
        """
        p=self.belief
        self.ent = self.calc_entropy(p)
        ent = self.ent
        #self.renderMatrix(p, 'img1')
        #self.renderMatrix(ent, 'img2')        
        p = (p - .5) * 2
        ent /= -np.log(.5)
        ent = (ent - .5) * 2



        if not single_agent:
            self.uuv_state_pos[0:3]=self.agentDispatcher.uuv.pose.pose_matrix[:3,3]       
            self.position2_5D[self.uuv_last_poseX,self.uuv_last_poseY]=-2        
            self.uuv_last_poseX,self.uuv_last_poseY =int(self.uuv_state_pos[0]), int(self.uuv_state_pos[1])
            if not self.done:
                self.position2_5D[self.uuv_last_poseX,self.uuv_last_poseY]=2*((self.uuv_state_pos[2]/ self.zn) - 0.5)
        
        self.uav_state_pos[0:3]=self.agentDispatcher.uav.pose.pose_matrix[:3,3]
        self.position2_5D[self.uav_last_poseX,self.uav_last_poseY]=-2    
        self.uav_last_poseX,self.uav_last_poseY =int(self.uav_state_pos[0]), int(self.uav_state_pos[1])
        if not self.done:
            self.position2_5D[self.uav_last_poseX,self.uav_last_poseY]=2*((self.uav_state_pos[2]/ self.zn) - 0.5)
        #self.renderMatrix(self.belief,name='ass')
        state=np.concatenate([np.expand_dims(2*((self.loaded_env.map_2_5D[:,:,0]/self.zn)-0.5),axis=-1), np.expand_dims(p, axis=-1)], axis=-1)
        state=np.concatenate((state,np.expand_dims(ent, axis=-1)), axis=-1)
        state=np.concatenate((state,np.expand_dims(self.position2_5D, axis=-1)), axis=-1)
        #self.renderMatrix(self.ent)
        return state


    def get_hLevel_observation(self):
        """calculates the high level map. It is implemented for a 2D case where we also return the 
        state of one agent (for RL).
        We store in one of the matrixes the obstacles and the position of the agent and in the second
        the entropy.
        
        TODO: optimise speed no need to copy the obstacles"""
        state_pose=np.copy(self.obstacles)
        self.uav_state_pos[0:2]=self.agentDispatcher.uav.pose.pose_matrix[:2,3]
        #self.uuv_state_pos[0:2]=self.agentDispatcher.uuv.pose.pose_matrix[:2,3]
        if self.done== False:
            state_pose[int(self.uav_state_pos[0]), int(self.uav_state_pos[1])]=2
        state=np.concatenate([np.expand_dims(self.entr_map,axis=-1), np.expand_dims(state_pose, axis=-1)], axis=-1)
        
        self.renderMatrix(state_pose)
        self.renderMatrix(self.entr_map, name="image2")


        return state


    def get_lLevel_observation(self):
        self.uav_state_pos[0:3]=self.agentDispatcher.uav.pose.pose_matrix[:3,3]
        self.sub_uav_pos[0:2]= self.uav_state_pos[0:2]%self.sub_env_size 
        self.sub_uav_pos[2]=self.uav_state_pos[2]
        x_hL,y_hL= int(self.uav_state_pos[0]//self.sub_env_size),  int(self.uav_state_pos[1]//self.sub_env_size)
        s_sub_env_x, s_sub_env_y = x_hL*self.sub_env_size, y_hL*self.sub_env_size
        e_sub_env_x, e_sub_env_y = (x_hL+1*self.sub_env_size)-1, (y_hL+1*self.sub_env_size)-1
        self.sub_ent=self.entr_map[s_sub_env_x:e_sub_env_x,s_sub_env_y:e_sub_env_y]
        self.sub_bel=self.belief[s_sub_env_x:e_sub_env_x,s_sub_env_y:e_sub_env_y]
        self.sub_map_2_5D=self.map_2_5D[s_sub_env_x:e_sub_env_x,s_sub_env_y:e_sub_env_y]

        return self.sub_uav_pos, entropy, belief

    
    def deterministic_rewards(self, entr_new, belief): 
            reward_map_entr=np.copy(self.reward_map_entr)
            reward_map_bel=np.copy(self.reward_map_bel)
            self.reward_map_entr[entr_new<.001]=1
            self.reward_map_bel[belief>.999]=1
            
            reward=0.0
            if self.done==False:
                reward=(np.sum(self.reward_map_bel)-np.sum(reward_map_bel))+((np.sum(self.reward_map_entr)-np.sum(reward_map_entr))/100)
            del reward_map_entr
            del reward_map_bel

            return reward

 
    def calc_reward(self, belief):
        entr_new=  self.calc_entropy(belief)
        if not self.done:
            self.reward= self.deterministic_rewards(entr_new, belief)
        self.litter= np.sum(self.reward_map_bel)            
        #if self.done:
        #    self.litter_amount=self.litter/np.sum(self.real_2_D_map[:,:,1])
        return np.sum(entr_new)
    
    def step(self, a, all_actions=False, h_level=True, agent=""):
        self.t += 1
        self.reward = 0.0
        self.done = False

        if(all_actions):
            a=self.agentDispatcher.get_legalMaxValAction(a)
        
        if False:
            self.entr_map, self.reward, self.done = self.agentDispatcher.greedy_multiagent_action(self.entr_map,a,3)
        else:
            #self._entr_map = np.where(self.entr_map<=0, 0, self.entr_map)
            #self.entr_map, self.reward, self.done = self.agentDispatcher.act(self.entr_map,a,h_level)
            belief, self.reward, self.done = self.agentDispatcher.act(self.belief,h_level,a)
        if self.t >= 400:#self.episode_length:
            self.timeout = True  
        if agent=="rainbow" or agent=="PPO":
            entr_new=self.calc_reward(belief)
        self.belief=belief
        if h_level:
            return self.get_hLevel_observation(), self.reward, self.done, self.timeout#np.sum(entr_new)
        else:
            return self.get_observation(), self.reward, self.done, self.timeout

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False