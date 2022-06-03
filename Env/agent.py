from Env.sensors import BayesianSensor
from Env.pytransform3d_.rotations import *
from Env.pytransform3d_.transformations import rotate_transform, transform_from, translate_transform
from Env.lookUpTable_actions import Actions
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import ctypes as c
import json
import time


class Pose:
    def __init__(self):
        self.init_R= matrix_from_axis_angle([0,0,1,-np.pi/2])
        self.pose_matrix=None

    def reset(self, x=0, y=0, z=0, orientation=[0,0,1,np.pi/2]):
        self.pose_matrix= transform_from(self.init_R,[x,y,z])
        self.Sim_pose_matrix= transform_from(self.init_R,[x,y,z])

class AgentDispatcher():
    def __init__(self, config, xn,yn,zn):
        self.uav=Agent(config, "UAV", xn, yn, zn)
        self.uuv=Agent(config, "UUV", xn, yn, zn)
        self.agent_list={self.uav,self.uuv}
        self.update_map ={}
        self.belief=np.zeros(xn ,yn)
        self.xn, self.yn = xn ,yn
    
    def reset(self, loaded_env):
        
        self.uav.reset_agent(loaded_env)
        self.uuv.reset_agent(loaded_env)

    def start_process_uav(self, action, belief, update_map, counter, uav_pose_matrix, uav_sensor_matrix):
        np_update_map=np.frombuffer(update_map.get_obj(),c.c_double)
        arr = np.frombuffer(belief.get_obj(),c.c_double)
        np_shared_belief = arr.reshape((self.xn, self.yn))

        UAVpose = np.frombuffer(uav_pose_matrix.get_obj(),c.c_double)
        np_pose_matrix = UAVpose.reshape(self.uav.pose.pose_matrix.shape)

        UAVsensor_matrix = np.frombuffer(uav_sensor_matrix.get_obj(),c.c_double)
        np_sensor_matrix = UAVsensor_matrix.reshape(self.uav.sensor_model.sensor_matrix.shape)   

        self.uav.make_action(action,np_pose_matrix, np_sensor_matrix, lm=False)
        self.uav.sensor_model.readSonarData(np_shared_belief, np_update_map,counter, sens_steps=5, simulate=False)  


    def start_process_uuv(self, action, belief, update_map, counter, uuv_pose_matrix, uuv_sensor_matrix):
        np_update_map=np.frombuffer(update_map.get_obj(),c.c_double)
        arr = np.frombuffer(belief.get_obj(),c.c_double)
        np_shared_belief = arr.reshape(self.xn, self.yn)

        UUVpose = np.frombuffer(uuv_pose_matrix.get_obj(),c.c_double)
        np_pose_matrix = UUVpose.reshape(self.uuv.pose.pose_matrix.shape)

        UUVsensor_matrix = np.frombuffer(uuv_sensor_matrix.get_obj(),c.c_double)
        np_sensor_matrix = UUVsensor_matrix.reshape(self.uuv.sensor_model.sensor_matrix.shape)


        self.uuv.make_action(action, np_pose_matrix, np_sensor_matrix, lm=False)
        self.uuv.sensor_model.readSonarData(np_shared_belief, np_update_map,counter, sens_steps=5, simulate=False)
        

    def multiprocess_action(self, belief, uav_action=0, uuv_action=0, lm=False):
        uav_action=0
        uuv_action=2
        self.belief=belief
        uuv_sensor_shape=self.uuv.sensor_model.sensor_shape
        uav_sensor_shape=self.uav.sensor_model.sensor_shape
        with mp.Manager() as manager:
            arr=np.zeros(uav_sensor_shape[0]*uuv_sensor_shape[1]+uav_sensor_shape[0]*uav_sensor_shape[1])
            update_map = mp.Array('d',arr,lock=True)
            counter = mp.Value('i',0, lock=True)
            shared_belief= mp.Array('d', belief.flatten("C"), lock=True)

            uav_pose_matrix= mp.Array('d', self.uav.pose.pose_matrix.flatten("C"), lock=True)
            uuv_pose_matrix= mp.Array('d', self.uuv.pose.pose_matrix.flatten("C"), lock=True)
            uav_sensor_matrix= mp.Array('d', self.uav.sensor_model.sensor_matrix.flatten("C"), lock=True)
            uuv_sensor_matrix= mp.Array('d', self.uuv.sensor_model.sensor_matrix.flatten("C"), lock=True)
            start=time.time()
            p1=mp.Process(target=self.start_process_uav,args=(uav_action, shared_belief, update_map, counter, uav_pose_matrix, uav_sensor_matrix))
            p2=mp.Process(target=self.start_process_uuv,args=(uuv_action, shared_belief, update_map, counter, uuv_pose_matrix, uuv_sensor_matrix))       
            p1.start()
            p2.start()
            p1.join()
            p2.join()
            end = time.time()
            print(end - start, 'action')
            shared_belief = np.frombuffer(shared_belief.get_obj(),c.c_double)
            belief = shared_belief.reshape(belief.shape)
            self.update_map = np.frombuffer(update_map.get_obj(),c.c_double)

            
            UAVpose_matrix = np.frombuffer(uav_pose_matrix.get_obj(),c.c_double)   
            self.uav.pose.pose_matrix = UAVpose_matrix.reshape(self.uav.pose.pose_matrix.shape)
            UUVpose_matrix = np.frombuffer(uuv_pose_matrix.get_obj(),c.c_double) 
            self.uuv.pose.pose_matrix = UUVpose_matrix.reshape(self.uuv.pose.pose_matrix.shape)
            UAVsensor_matrix = np.frombuffer(uav_sensor_matrix.get_obj(),c.c_double)   
            self.uav.sensor_model.sensor_matrix = UAVsensor_matrix.reshape(self.uav.sensor_model.sensor_matrix.shape)
            UUVsensor_matrix = np.frombuffer(uuv_sensor_matrix.get_obj(),c.c_double) 
            self.uuv.sensor_model.sensor_matrix = UUVsensor_matrix.reshape(self.uuv.sensor_model.sensor_matrix.shape)   
            
        return belief

    def simple_multiagent_action(self, entropy, uav_action, uuv_action):
        done=self.uav.make_action(uav_action, self.uav.pose.pose_matrix, self.uav.sensor_model.sensor_matrix)
        #self.uuv.make_action(uuv_action, self.uuv.pose.pose_matrix, self.uuv.sensor_model.sensor_matrix)
        #print(int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3]))
        
        #reward_uuv=entropy[int(self.uuv.pose.pose_matrix[0,3]),int(self.uuv.pose.pose_matrix[1,3])]-0.2*entropy[int(self.uuv.pose.pose_matrix[0,3]),int(self.uuv.pose.pose_matrix[1,3])]
        
        #entropy[int(self.uuv.pose.pose_matrix[0,3]),int(self.uuv.pose.pose_matrix[1,3])]= 0.2*entropy[int(self.uuv.pose.pose_matrix[0,3]),int(self.uuv.pose.pose_matrix[1,3])]
        if done:
            reward_uav=-1
        else:
            reward_uav=entropy[int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3])]-0.2*entropy[int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3])]
            entropy[int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3])]= 0.2*entropy[int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3])]
        return entropy, reward_uav, done


    def greedy_multiagent_chose_action(self):
        action_new=self.uav.sim_greedy(uav_action, self.uav.pose.pose_matrix, self.uav.sensor_model.sensor_matrix)
        return action_new

    def sim_greedy(self):
        self.uav.sim_greedy()
        self.uuv.sim_greedy()
    
#    def read_sensors(self,belief):
#        self.update_map.clear()
#        belief, self.update_map = self.uav.sensor_model.readSonarData(belief, self.update_map, sens_steps=5, simulate=False)
#        belief, self.update_map = self.uuv.sensor_model.readSonarData(belief, self.update_map, sens_steps=5, simulate=False)
#        return belief, self.update_map

    def init_render_sensors(self, fig):
        fig, uav = self.uav.sensor_model.init_render_sensor(fig)
        fig, uuv = self.uuv.sensor_model.init_render_sensor(fig)
        return fig, uav, uuv
    
    def render_sensors(self, uav, uuv):
        uav_beams = self.uav.sensor_model.render(uav)
        uuv_beams = self.uuv.sensor_model.render(uuv)
        return uav_beams, uuv_beams

class Agent():
    def __init__(self,config, agent_name, xn,yn,zn):
        self.xn, self.yn, self.zn = xn, yn, zn
        self.agent_key=agent_name
        self.config=config
        self.random_pose=config.getboolean(self.agent_key,"random_pose")
        rot_speed_degr=config.getint(self.agent_key,"rot_speed_degr")
        self.speed=config.getint(self.agent_key,"speed")
        init_pose_json=config.get(self.agent_key,'init_pose')
        init_pose=json.loads(init_pose_json)
        self.init_pose=np.array(init_pose)

        self.pose = Pose()
        self.sensor_model = BayesianSensor(self.config, self.agent_key, self.pose)

        self.last_action=0
        self.last_poseX=0
        self.last_poseY=0
        self.rad=np.deg2rad(rot_speed_degr)
        self.actions= Actions(self.rad)


    def reset_agent(self, loaded_env):
        self.env_min = loaded_env._2_D_min
        self.hash_map = loaded_env.hashmap
        self.real_2_D_map = loaded_env.map_2_5D
        self.map = loaded_env.voxel_map
        self.hashmap =  loaded_env.hashmap
        if self.random_pose:
            self.x0, self.y0 = np.random.randint(1, self.xn-2), np.random.randint(1, self.yn-2)
            min_z = self.real_2_D_map[self.x0][self.y0][0]
            assert min_z!=self.zn or min_z+1!=self.zn or min_z+1!=self.zn-1

            self.z0= 11 #np.random.randint((min_z+1), (self.zn/2-2))
            self.rotation=random_axis_angle()
        else:
            self.x0, self.y0, self.z0 = self.init_pose[0], self.init_pose[1], self.init_pose[2]
            self.rotation=random_axis_angle()
        self.pose.reset(x=self.x0, y=self.y0, z=self.z0)
        self.sensor_model.reset(self.map, self.pose, self.hashmap)


    def sim_greedy(self):
        action_new=0
        H_old=0
        R_t=np.eye(4)
        for action in range(8):#self.env.ACTIONS.shape[0]):
            new_position=self.pose.pose_matrix[:3,3] + self.actions.ACTIONS[action][:3]
            if self.legal_change_in_pose(new_position):
                #if(action%7==1 or action%7==2):
                #    R_t=matrix_from_axis_angle([1, 0, 0, self.actions.ACTIONS[action][3]])
                #elif(action%7==3 or action%7==4):
                #    R_t=matrix_from_axis_angle([0, 1, 0, self.actions.ACTIONS[action][4]])
                #elif(action%7==5 or action%7==6):
                #    R_t=matrix_from_axis_angle([0, 0, 1, self.actions.ACTIONS[action][5]])  
                self.sensor_model.Sim_sensor_matrix=self.sensor_model.sensor_matrix.copy()
                self.sensor_model.Sim_sensor_matrix[:,:,:3,3]=new_position
                self.pose.Sim_pose_matrix=np.matmul(self.pose.pose_matrix[:3,:3],R_t[:3,:3]) 
                self.sensor_model.Sim_sensor_matrix[:,:,:3,:3]= np.matmul(self.pose.Sim_pose_matrix[:3,:3],self.sensor_model.sensor_matrix_init[:,:,:3,:3])
                self.Sim_legal_action=True
                H, B=self.sensor_model.readSonarData(sens_steps=5, simulate=True)
                if B>B_old:
        
                    B_old=B
                    action_new=action

        return action_new
        #self.make_action(action_new, simulate=False)

    def make_action(self, action, np_pose_matrix, np_sensor_matrix=None, lm=False):
        R_t=np.eye(4)
        if lm==False:
            new_position=np_pose_matrix[:3,3] + self.actions.ACTIONS[action][:3]
            if self.legal_change_in_pose(new_position):
                ## np.matmul(self.pose.pose_matrix[:3,:3],self.ACTIONS[a][:3])###<-------s
                #if(action%7==1 or action%7==2):
                #    R_t=matrix_from_axis_angle([1, 0, 0,self.ACTIONS[action][3]])
                #elif(action%7==3 or action%7==4):
                #    R_t=matrix_from_axis_angle([0, 1, 0,self.ACTIONS[action][4]])
                #elif(action%7==5 or action%7==6):
                #    R_t=matrix_from_axis_angle([0, 0, 1,self.ACTIONS[action][5]])  
                
                #self.sensor_model.sensor_matrix=self.sensor_model.sensor_matrix.copy()
                np_pose_matrix[:3,3]= new_position
                np_pose_matrix[:3,:3]= np.matmul(np_pose_matrix[:3,:3],R_t[:3,:3])
                np_sensor_matrix[:,:,:3,3]= new_position
                np_sensor_matrix[:,:,:3,:3]= np.matmul(np_pose_matrix[:3,:3],self.sensor_model.sensor_matrix_init[:,:,:3,:3])
                return False
            else:
                return True

        elif lm==True:
            if a<6:
                new_position=self.pose.pose_matrix[:3,3] + self.actions.ACTIONS[a][:3]# np.matmul(self.pose.pose_matrix[:3,:3],self.ACTIONS[a][:3])###<-------s
                if not self.legal_change_in_pose(new_position):
                    self.done=True
    
    
    def in_map(self, new_pos):
        return new_pos[0] >= 0 and new_pos[1] >= 0 and new_pos[0] <= (self.xn-1) and new_pos[1] <= (self.yn-1) and new_pos[2] >= 0 and new_pos[2] <= (self.zn/2)+2#(self.zn/2-1)

    def _2Dcollision(self, new_pos):
        if self.real_2_D_map[int(new_pos[0]),int(new_pos[1]),0]>=new_pos[2]:
            #print("COLLISION ! ! !", new_pos[0],int(new_pos[1]),int(new_pos[2]))
            return True
        else:
            return False


    def collision(self,new_pos):
        x = int(np.rint(new_pos[0]))
        y = int(np.rint(new_pos[1]))
        z = int(np.rint(new_pos[2]))
        hashkey = 1000000*x+1000*y+z
        if hashkey in self.hashmap:
            #print("COLLISION ! ! !", x,y,z)
            return True
        else:
            return False

    def legal_rotation(self, new_pos):
        angleX,angleY, _ = (180/math.pi)*euler_xyz_from_matrix(new_pos)
        if (abs(angleX)>=91) or (abs(angleY)>=91):
            return False
        else:
            return True


    def legal_change_in_pose(self,new_position): 
        if self.in_map(new_position) and not self._2Dcollision(new_position):#self.collision(new_position):
            return True
        else:
            return False
    