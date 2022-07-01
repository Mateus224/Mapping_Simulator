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


def safe_log(x):
    if x <= 0.:
        return 0.
    return np.log(x)
safe_log = np.vectorize(safe_log)

class Pose:
    def __init__(self):
        self.init_R= matrix_from_axis_angle([0,0,1,-np.pi/2])
        self.pose_matrix=None

    def reset(self, x=0, y=0, z=0, orientation=[0,0,1,np.pi/2]):
        self.pose_matrix= transform_from(self.init_R,[x,y,z])
        self.Sim_pose_matrix= transform_from(self.init_R,[x,y,z])

class AgentDispatcher():
    def __init__(self, config, xn,yn,zn):
        self.b_multiagent=False
        self.uav=Agent(config, "UAV", xn, yn, zn)
        #self.uuv=Agent(config, "UUV", xn, yn, zn)
        #self.agent_list={self.uav,self.uuv}
        self.update_map ={}
        self.belief=np.zeros(xn ,yn)
        self.xn, self.yn = xn ,yn
    
    def reset(self, loaded_env):
        
        self.uav.reset_agent(loaded_env)
        #self.uuv.reset_agent(loaded_env)

    def start_process_uav(self, action, belief, update_map, counter, uav_pose_matrix, uav_sensor_matrix):
        np_update_map=np.frombuffer(update_map.get_obj(),c.c_double)
        arr = np.frombuffer(belief.get_obj(),c.c_double)
        np_shared_belief = arr.reshape((self.xn, self.yn))

        UAVpose = np.frombuffer(uav_pose_matrix.get_obj(),c.c_double)
        np_pose_matrix = UAVpose.reshape(self.uav.pose.pose_matrix.shape)

        UAVsensor_matrix = np.frombuffer(uav_sensor_matrix.get_obj(),c.c_double)
        np_sensor_matrix = UAVsensor_matrix.reshape(self.uav.sensor_model.sensor_matrix.shape)   

        self.uav.make_action(action,np_pose_matrix, np_sensor_matrix)
        self.uav.sensor_model.readSonarData(np_shared_belief, np_update_map,counter, sens_steps=5, simulate=False)  


    def start_process_uuv(self, action, belief, update_map, counter, uuv_pose_matrix, uuv_sensor_matrix):
        np_update_map=np.frombuffer(update_map.get_obj(),c.c_double)
        arr = np.frombuffer(belief.get_obj(),c.c_double)
        np_shared_belief = arr.reshape(self.xn, self.yn)

        UUVpose = np.frombuffer(uuv_pose_matrix.get_obj(),c.c_double)
        np_pose_matrix = UUVpose.reshape(self.uuv.pose.pose_matrix.shape)

        UUVsensor_matrix = np.frombuffer(uuv_sensor_matrix.get_obj(),c.c_double)
        np_sensor_matrix = UUVsensor_matrix.reshape(self.uuv.sensor_model.sensor_matrix.shape)


        self.uuv.make_action(action, np_pose_matrix, np_sensor_matrix)
        self.uuv.sensor_model.readSonarData(np_shared_belief, np_update_map,counter, sens_steps=5, simulate=False)
        

    def multiprocess_action(self, belief, uav_action=0, uuv_action=0):
        """Is implemented just for the low level layer because it makes no sense to use it on the fast high level layer"""
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
            #print(end - start, 'action')
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

            ##define rewardfunction if needed reward has to be an 2x1 array##
            reward= [0,0]
            done=False
            
        return belief, reward, done

    def singleprocess_action(self, belief, uav_action, uuv_action, h_level=True):
        #done=self.uav.make_action(uav_action, self.uav.pose.pose_matrix, self.uav.sensor_model.sensor_matrix)
        a_notL=-1
        if not self.sim_legal_action(uav_action):
            a_n=np.random.random_integers(7)
            done=False
            a_notL=uav_action
            while not done:
                if self.sim_legal_action(a_n) :
                    uav_action=a_n
                    done=True
                else:
                    a_n=np.random.random_integers(7)
        done=self.uav.make_action(uav_action, self.uav.pose.pose_matrix, self.uav.sensor_model.sensor_matrix)
        if done:
            print(done,"!!!!")



        #    self.sim_legal_action(action):
 
        #entropy=self.calc_entropy(belief)
        #self.uuv.make_action(uuv_action, self.uuv.pose.pose_matrix, self.uuv.sensor_model.sensor_matrix)
        #print(int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3]))
        
        #reward_uuv=entropy[int(self.uuv.pose.pose_matrix[0,3]),int(self.uuv.pose.pose_matrix[1,3])]-0.2*entropy[int(self.uuv.pose.pose_matrix[0,3]),int(self.uuv.pose.pose_matrix[1,3])]
        
        #entropy[int(self.uuv.pose.pose_matrix[0,3]),int(self.uuv.pose.pose_matrix[1,3])]= 0.2*entropy[int(self.uuv.pose.pose_matrix[0,3]),int(self.uuv.pose.pose_matrix[1,3])]
        #if done:
        #    reward_uav=-0.35
        else:
            if h_level==True:
                
                reward_uav=entropy[int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3])]-0.2*entropy[int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3])]
                entropy[int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3])]= 0.2*entropy[int(self.uav.pose.pose_matrix[0,3]),int(self.uav.pose.pose_matrix[1,3])]
            else:
                belief, _ = self.uav.sensor_model.readSonarData(belief, self.update_map, 0)
                #print(self.update_map)
                reward_uav=None
        return belief, a_notL, uav_action, reward_uav, done
    
    def act(self, belief, h_level,uav_action,  uuv_action=None, multiprocess=False):
        if multiprocess:
            belief, reward_uav, done=self.multiprocess_action(belief, uav_action=0, uuv_action=0)
        else:
            belief, a_notL, a, reward_uav, done=self.singleprocess_action(belief, uav_action, uuv_action, h_level)
        return belief, a_notL,a, reward_uav, done


    def get_legalMaxValAction(self, actions):
        if self.b_multiagent:
            pass
        else:
            action=self.uav.sim_actions(actions)
        return action

    def greedy_multiagent_chose_action(self, belief):
        action_new=self.sim_greedy(belief)
        return action_new

    def sim_greedy(self, belief):
        uav_action=self.uav.sim_greedy(belief, self.update_map)
        #uuv_action=self.uuv.sim_greedy(belief, self.update_map)
        return uav_action #, uuv_action
    
#    def read_sensors(self,belief):
#        self.update_map.clear()
#        belief, self.update_map = self.uav.sensor_model.readSonarData(belief, self.update_map, sens_steps=5, simulate=False)
#        belief, self.update_map = self.uuv.sensor_model.readSonarData(belief, self.update_map, sens_steps=5, simulate=False)
#        return belief, self.update_map

    def sim_legal_action(self, action):
        if self.b_multiagent:
            pass
        valid=self.uav.sim_legal_action(action)
        return valid

    def calc_entropy(self, b_t):
        entropy = - (b_t * safe_log(b_t) + (1 - b_t) * safe_log(1 - b_t))

        return entropy


    def init_render_sensors(self, fig, b_multiagent):
        self.b_multiagent=b_multiagent
        if b_multiagent:
            fig, uav = self.uav.sensor_model.init_render_sensor(fig)
            fig, uuv = self.uuv.sensor_model.init_render_sensor(fig)
        else:
            fig, uav = self.uav.sensor_model.init_render_sensor(fig)
            uuv=None
        return fig, uav, uuv
    
    def render_sensors(self, uav, uuv):
        if self.b_multiagent:
            uav_beams = self.uav.sensor_model.render(uav)
            uuv_beams = self.uuv.sensor_model.render(uuv)
        else:
            uav_beams = self.uav.sensor_model.render(uav)
            uuv_beams=None
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

        self.action_space=0
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

            self.z0= 14 #np.random.randint((min_z+1), (self.zn/2-2))
            self.rotation=random_axis_angle()
        else:
            self.x0, self.y0, self.z0 = self.init_pose[0], self.init_pose[1], self.init_pose[2]
            self.rotation=random_axis_angle()
        self.pose.reset(x=self.x0, y=self.y0, z=self.z0)
        self.sensor_model.reset(self.map, self.pose, self.hashmap)

    def sim_legal_action(self,action):
        if self.action_space==0:
            action_set = self.actions.ACTIONS2D
        elif action_space==1:
            action_set = self.actions.ACTIONS3D
        new_position=self.pose.pose_matrix[:3,3] + action_set[action][:3]
        legal_pos=self.legal_change_in_pose(new_position,_2D=False)
        return legal_pos

    def sim_actions(self,actionArr):
        if self.action_space==0:
            action_set = self.actions.ACTIONS2D
        elif action_space==1:
            action_set = self.actions.ACTIONS3D
        not_chosen=True
        while not_chosen:
            action=np.argmax(actionArr)
            new_position=self.pose.pose_matrix[:3,3] + action_set[action][:3]
            if self.legal_change_in_pose(new_position):
                not_chosen=False
            else:
                actionArr[action]=0
        return action
        
                



    def sim_greedy(self, belief, update_map, action_space=0, sub_map_border=None):
        """Simulates all possible actions and takes a greedy action
        TODO: multi process

        PARAMETERS
        ----------
        
        action_space: defines which action set is takes from the Action Class."""
        action_new=0
        B_old=0
        R_t=np.eye(4)
        if action_space==0:
            action_set = self.actions.ACTIONS2D
        elif action_space==1:
            action_set = self.actions.ACTIONS3D
        for action in action_set[0]:
            new_position=self.pose.pose_matrix[:3,3] + action_set[action][:3]
            if self.legal_change_in_pose(new_position,sub_map_border):
                R_t=np.matmul(np.matmul(matrix_from_axis_angle([1, 0, 0, action_set[action][3]]), \
                    matrix_from_axis_angle([0, 1, 0, action_set[action][4]])), \
                    matrix_from_axis_angle([0, 0, 1, action_set[action][5]]))

                self.sensor_model.Sim_sensor_matrix=self.sensor_model.sensor_matrix.copy()
                self.sensor_model.Sim_sensor_matrix[:,:,:3,3]=new_position
                self.pose.Sim_pose_matrix=np.matmul(self.pose.pose_matrix[:3,:3],R_t[:3,:3]) 
                self.sensor_model.Sim_sensor_matrix[:,:,:3,:3]= np.matmul(self.pose.Sim_pose_matrix[:3,:3],self.sensor_model.sensor_matrix_init[:,:,:3,:3])
                self.Sim_legal_action=True
                H, B=self.sensor_model.readSonarData( belief, update_map, 0, sens_steps=5, simulate=True)
                if B>B_old:
        
                    B_old=B
                    action_new=action

        return action_new
        #self.make_action(action_new, simulate=False)

    def make_action(self, action, np_pose_matrix, np_sensor_matrix=None, action_space=0):        
        """Is implemented for multiprocess if np_sensor_matrix is not None we store there the new matrix

        PARAMETERS
        ----------

        np_sensor_matrix: we store there the sesnor matrix in case we are using multiprozesses
        action_space: defines which action set is takes from the Action Class.
        """
        R_t=np.eye(4)
        if action_space==0:
            action_set = self.actions.ACTIONS2D
        elif action_space==1:
            action_set = self.actions.ACTIONS3D
        new_position=np_pose_matrix[:3,3] + action_set[action][:3]
        if self.legal_change_in_pose(new_position, _2D=False):
            R_t=np.matmul(np.matmul(matrix_from_axis_angle([1, 0, 0,action_set[action][3]]), \
                matrix_from_axis_angle([0, 1, 0, action_set[action][4]])), \
                matrix_from_axis_angle([0, 0, 1, action_set[action][5]]))
            ## np.matmul(self.pose.pose_matrix[:3,:3],self.ACTIONS[a][:3])###<---action in agents coordinates
            np_pose_matrix[:3,3]= new_position
            np_pose_matrix[:3,:3]= np.matmul(np_pose_matrix[:3,:3],R_t[:3,:3])
            np_sensor_matrix[:,:,:3,3]= new_position
            np_sensor_matrix[:,:,:3,:3]= np.matmul(np_pose_matrix[:3,:3],self.sensor_model.sensor_matrix_init[:,:,:3,:3])
            
            return False
        else:
            return True

        #elif lm==True:
        #    if a<6:
        #        new_position=self.pose.pose_matrix[:3,3] + self.actions.ACTIONS[a][:3]# np.matmul(self.pose.pose_matrix[:3,:3],self.ACTIONS[a][:3])###<-------s
        #        if not self.legal_change_in_pose(new_position):
        #            self.done=True
    
    
    def in_map(self, new_pos):
        return new_pos[0] >= 0 and new_pos[1] >= 0 and new_pos[0] <= (self.xn-1) and new_pos[1] <= (self.yn-1) and new_pos[2] >= 0 and new_pos[2] <= (self.zn/2)+3#(self.zn/2-1)

    def in_sub_map(self, new_pos,sub_map_border):
        """Checks if an agents action is still in the boarders of the underlying sub environment.
        
        Parameters
        ----------
        new_pos : numpy array
            The current position of the agent
            
        sub_map_border: numpy array
            The bparders of the underlying sub environment where the order is [sub_x0,sub_xn,sub_y0,sub_yn]
            """
        return new_pos[0] >= sub_map_border[0] and new_pos[1] >= sub_map_border[2] and new_pos[0] <= sub_map_border[1] and new_pos[1] <= sub_map_border[3]
    
    def _2Dcollision(self, new_pos):
        if self.real_2_D_map[int(new_pos[0]),int(new_pos[1]),0]>=new_pos[2]:
            #print("COLLISION ! ! !", new_pos[0],int(new_pos[1]),int(new_pos[2]))
            return True
        else:
            return False



    def no_collision(self,new_pos):
        x = int(np.rint(new_pos[0]))
        y = int(np.rint(new_pos[1]))
        z = int(np.rint(new_pos[2]))
        hashkey = 1000000*x+1000*y+z
        if hashkey in self.hashmap:
            print("COLLISION ! ! !", x,y,z)
            #print(self.real_2_D_map)
            return False
        else:
            return True

    def legal_rotation(self, new_pos):
        angleX,angleY, _ = (180/math.pi)*euler_xyz_from_matrix(new_pos)
        if (abs(angleX)>=91) or (abs(angleY)>=91):
            return False
        else:
            return True


    def legal_change_in_pose(self, new_position, sub_map_border=None, _2D=True): 
        in_sub_map=True
        if sub_map_border!=None:
            in_sub_map=self.in_sub_map(new_position, sub_map_border)
        if _2D:
            return self.in_map(new_position) and not self._2Dcollision(new_position) and in_sub_map
        else:
            return self.in_map(new_position) and self.no_collision(new_position) 
