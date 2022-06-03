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
        self._entr_map=np.zeros((self.xn,self.yn))
        self.t = None
        self.viewer = None
        self.ent = None
        self.episode=0
        self.agentDispatcher=AgentDispatcher(config, self.xn, self.yn, self.zn)
        self.start_entr_map=167
        self.uav_state_pos=np.zeros(7)
        self.uuv_state_pos=np.zeros(7)
        



    def reset(self, episode_length=750, validation=False):
        self.entr_map=self.calc_entropy(np.ones((self.xn, self.yn))/2)
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
        self.belief=np.ones((self.map.shape[0],self.map.shape[1]))/2
        self.tmp_coordinate_storage=np.zeros((self.env_shape[0],self.env_shape[1], self.env_shape[2]))
        self.reward_map_entr=np.zeros((self.env_shape[0],self.env_shape[1]))
        self.reward_map_bel=np.zeros((self.env_shape[0],self.env_shape[1]))

        self.agentDispatcher.reset(self.loaded_env)
        self.position2_5D=np.zeros((self.xn, self.yn))-2
        self.position2_5D_R=np.zeros((self.xn, self.yn))
        self.last_poseX,self.last_poseY= 0,0
        self.litter=0
        self.litter_amount=0

        self.entr_old = self.calc_entropy(np.ones((self.xn, self.yn))/2)
        self.t = 0
        self.litter=0
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
        self.entr_map=np.where(self.obstacles==1, -1, self.entr_map)
        self.entr_map=np.where(self.obstacles==1, 0, self.entr_map)
        self.loaded_env.map_2_5D[:,:,0]=np.where(self.obstacles==1, plane, 0)
        self.start_entr_map = np.sum(self.entr_map)
        #print(self.loaded_env.map_2_5D[:,:,0])

        return self.get_observation_() , self.map



    def calc_entropy(self, p_t):
        entropy = - (p_t * safe_log(p_t) + (1 - p_t) * safe_log(1 - p_t))

        return entropy

    def calc_sum_entropy(self, obs):
        np.sum(obs)
        return

    def observation_size(self):
        return 2 * self.N - 1

    def get_observation(self):
        p=self.belief
        
        self.ent = self.calc_entropy(p)
        ent = self.ent
        
        p = (p - .5) * 2
        ent /= -np.log(.5)
        ent = (ent - .5) * 2

        self.uav_state_pos[0:3]=self.agentDispatcher.uav.pose.pose_matrix[:3,3]
        self.uuv_state_pos[0:3]=self.agentDispatcher.uuv.pose.pose_matrix[:3,3]
        self.position2_5D[self.last_poseX,self.last_poseY]=-2
        self.auv_last_poseX,self.last_poseY =int(self.uav_state_pos[0]), int(self.uav_state_pos[1])
        self.uuv_last_poseX,self.last_poseY =int(self.uuv_state_pos[0]), int(self.uuv_state_pos[1])
        if not self.done:
            self.position2_5D[self.last_poseX,self.last_poseY]=2*((self.uav_state_pos[2]/ self.zn) - 0.5)
            self.position2_5D[self.last_poseX,self.last_poseY]=200*((self.uuv_state_pos[2]/ self.zn) - 0.5)
            #self.position2_5D_R[x-1:x+2,y-1:y+2]=self.pose.pose_matrix[:3,:3]
        stack=np.concatenate([np.expand_dims(2*((self.loaded_env.map_2_5D[:,:,0]/self.zn)-0.5),axis=-1), np.expand_dims(p, axis=-1)], axis=-1)
        belief=np.concatenate((stack,np.expand_dims(ent, axis=-1)), axis=-1)
        height=np.concatenate(belief,np.expand_dims(self.position2_5D, axis=-1), axis=-1)
        #state=np.concatenate((height,np.expand_dims(self.position2_5D_R, axis=-1)), axis=-1)
        state=height
        
        #test=self.belief*255   
        #entr= test.astype(np.uint8)
        #cv2.imshow('image',entr)
        #cv2.waitKey(1)
        return state

    def get_observation_(self):
        state_pose=np.copy(self.obstacles)
        self.uav_state_pos[0:2]=self.agentDispatcher.uav.pose.pose_matrix[:2,3]
        #self.uuv_state_pos[0:2]=self.agentDispatcher.uuv.pose.pose_matrix[:2,3]
        if self.done== False:
            state_pose[int(self.uav_state_pos[0]), int(self.uav_state_pos[1])]=2
        state=np.concatenate([np.expand_dims(self.entr_map,axis=-1), np.expand_dims(state_pose, axis=-1)], axis=-1)
        state_pose=state_pose*255   
        state_pose= state_pose.astype(np.uint8)
        cv2.imshow('image3',state_pose)

        entr=self._entr_map*255   
        entr= entr.astype(np.uint8)
        cv2.imshow('image',entr)
        cv2.waitKey(1)
        return state

    
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

 
    def calc_reward(self):
        entr_new=  self.calc_entropy(self.belief)
        if not self.done:
            self.reward= self.deterministic_rewards(entr_new, self.belief)
        self.litter= np.sum(self.reward_map_bel)            
        if self.t >= 800:#self.episode_length:
            self.done= True  
            self.t = 0
        if self.done:
            self.litter_amount=self.litter/np.sum(self.real_2_D_map[:,:,1])
        return np.sum(entr_new)
    
    def step(self, a, lm=False, agent=""):
        self.t += 1
        self.reward = 0.0
        self.done = False
        if False:
            self.entr_map, self.reward, self.done = self.agentDispatcher.greedy_multiagent_action(self.entr_map,a,3)
        else:
            self._entr_map = np.where(self.entr_map<=0, 0, self.entr_map)
            #self.belief = self.agentDispatcher.multiprocess_action( self.belief,a, lm=lm)
            self.entr_map, self.reward, self.done = self.agentDispatcher.simple_multiagent_action(self._entr_map,a,3)
        if self.t >= 700:#self.episode_length:
            self.done= True  
        #if agent=="DDDQN" or agent=="PPO":
        #    entr_new=self.calc_reward()
        
        return self.get_observation_(), self.reward, self.done, None#np.sum(entr_new)

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False