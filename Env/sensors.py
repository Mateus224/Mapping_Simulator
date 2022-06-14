from Env.pytransform3d_.rotations import *
import numpy as np
import random
import json



def safe_log(x):
    if x <= 0.:
        return 0.
    return np.log(x)
safe_log = np.vectorize(safe_log)


class BayesianSensor():
    def __init__(self, config, agent_key, pose):
        
        sensor_shape_json=config.get(agent_key,'sensor_shape')
        sensor_shape=json.loads(sensor_shape_json)
        self.sensor_shape=np.array(sensor_shape)

        sensor_reflection_json=config.get(agent_key,'sensor_reflection')
        sensor_reflection=json.loads(sensor_reflection_json)
        self.sensor_reflection=np.array(sensor_reflection)

        self.sensor_range=config.getint(agent_key,"sensor_range")
        self.opening_angle=config.getint(agent_key,"opening_angle")
        self.beam_width=config.getint(agent_key,"beam_width")
        self.diastance_accuracy=config.getfloat(agent_key,"diastance_accuracy")

        self.num_beams=sensor_shape[0]
        self.num_rays=sensor_shape[1]
        self.beamWidth = (self.beam_width*np.pi)/180 # convert into radiant
        self.separation =  self.beamWidth/self.num_beams
        self.beamOpening=(self.opening_angle*np.pi)/180 # convert into radiant

        self.pose=pose
        self.h=0
        self.b=0
        self.update_map={}
        
        self.sensor_matrix=np.zeros((self.num_beams, int(self.sensor_reflection.shape[0]), 4,4))
        self.Sim_sensor_matrix=np.zeros((self.num_beams, int(self.sensor_reflection.shape[0]), 4,4))
        self.update_sensor_matrix=np.zeros((self.num_beams, int(self.sensor_reflection.shape[0])))
        self.pose.poseSim_matrix=None


    def init_sensor(self):
        self.R_render=self.rotation_matrix_from_vectors([1,1,1], [0,0,-1])
        self.sensor_matrix[:,:,:,3]=self.pose.pose_matrix[:, 3]
        A2C=np.eye(3)
        for i in range(self.num_beams):
            for j in range(self.sensor_reflection.shape[0]):
                if .5*self.num_beams < i:

                    self.sensor_matrix[i,j,:3, :3] = np.matmul(self.pose.pose_matrix[:3, :3], matrix_from_axis_angle([1, 0, 0, -(self.num_beams-i)*self.separation])) 
                    if.5*self.sensor_reflection.shape[0] < j: 
                        self.sensor_matrix[i,j,:3, :3]=np.matmul(self.sensor_matrix[i,j,:3, :3], matrix_from_axis_angle([0, 1, 0, -(self.sensor_reflection.shape[0]-j)*self.beamOpening/self.sensor_reflection.shape[0]])) 
                    else:
                        self.sensor_matrix[i,j,:3, :3] = np.matmul(self.sensor_matrix[i,j,:3, :3], matrix_from_axis_angle([0, 1, 0, j*self.beamOpening/self.sensor_reflection.shape[0]]))
                else:
                    self.sensor_matrix[i,j,:3, :3] = np.matmul(self.pose.pose_matrix[:3, :3], matrix_from_axis_angle([1, 0, 0, i*self.separation])) 
                    if.5*self.sensor_reflection.shape[0] < j: 
                        self.sensor_matrix[i,j,:3, :3]=np.matmul( self.sensor_matrix[i,j,:3, :3],matrix_from_axis_angle([0, 1, 0, -(self.sensor_reflection.shape[0]-j)*self.beamOpening/self.sensor_reflection.shape[0]])) 
                    else:
                        self.sensor_matrix[i,j,:3, :3] = np.matmul(self.sensor_matrix[i,j,:3, :3],matrix_from_axis_angle([0, 1, 0, j*self.beamOpening/self.sensor_reflection.shape[0]]))
        self.sensor_matrix_init=self.sensor_matrix.copy()


    def reset(self, map, pose, hashmap):
        self.map = map
        self.pose=pose
        self.hashmap=hashmap
        self.init_sensor()
        

    
    def readSonarData(self, belief, update_map, counter, sens_steps=3, simulate=False):
        self.h=0
        self.b=0  
        self.update_map.clear() 
        doneVoxel=1
        hit= np.zeros([self.sensor_matrix.shape[0],self.sensor_matrix.shape[1]])
        for z_ in range(self.sensor_range*sens_steps):
            ray_z=z_/sens_steps
            if simulate==True:
                measurments=self.Sim_sensor_matrix[:,:,:3,:3].dot([0,0,-ray_z])
                measurments= measurments+self.Sim_sensor_matrix[:,:,:3,3]         
            else:
                measurments=self.sensor_matrix[:,:,:3,:3].dot([0,0,-ray_z])
                measurments= measurments+self.sensor_matrix[:,:,:3,3]
            for i in range (measurments.shape[0]):
                for j in range (measurments.shape[1]):
                    if (hit[i,j]==0):
                        hashkey = 1000000*int(np.rint(measurments[i,j,0]))+1000*int(np.rint(measurments[i,j,1]))+int(np.rint(measurments[i,j,2]))
                        hashkey_zp1 = 1000000*int(np.rint(measurments[i,j,0]))+1000*int(np.rint(measurments[i,j,1]))+int(np.rint(measurments[i,j,2])+1)
                        
                        if hashkey_zp1 in self.hashmap:
                            hit[i,j]=1
                        else:
                            if hashkey in self.hashmap:
                                hit[i,j]=1
                                if hashkey in update_map:
                                    doneVoxel=doneVoxel+1
                                else:
                                    if type(counter) == int: #for single agent
                                        update_map[counter]=hashkey
                                        counter +=1   
                                    else:                     #for multi process agents                
                                        update_map[counter.value]=hashkey
                                        counter.value +=1
                                    #print(self.update_map[hashkey])
                                #if test==True:
                                #    continue
                                
                                value=self.hashmap.get(hashkey)
                                x=int(np.rint(measurments[i,j,0]))
                                y=int(np.rint(measurments[i,j,1]))
                                if simulate!=True:
                                    self.updateBelief(belief,value,ray_z,x,y,j)
                                else:
                                    entropy = - (belief[x,y] * safe_log(belief[x,y]) + (1 - belief[x,y]) * safe_log(1 - belief[x,y]))
                                    self.h=self.h+np.abs(entropy)
                                    if (belief[x,y]<0.999 ):
                                        b=self.b+(doneVoxel*belief[x,y])
        if simulate!=True:
            return belief, update_map
        else:
            return self.h, self.b




    def updateBelief(self,belief, value,ray_z,x,y,j):
        p_z = np.power(self.diastance_accuracy,ray_z)*self.sensor_reflection[j]
        if random.random()<=p_z:
            if value==0.5:
                belief[x,y]=(p_z *belief[x,y]) / ((p_z * belief[x,y]) + ((1-p_z)*(1-belief[x,y])))
            else:
                belief[x,y]=1-(((p_z) *(1-belief[x,y])) / (((p_z) * (1-belief[x,y]) + ((1-p_z)*(belief[x,y])))))
        else:
            if value==0.5:
                belief[x,y]=((1-p_z) *(belief[x,y]))/ (((1-p_z) * belief[x,y]) + (p_z*(1-belief[x,y])))
            else:   
                belief[x,y]=1-(((1-p_z) *(1-belief[x,y]))/ (((1-p_z) * (1-belief[x,y])) + ((p_z)*(belief[x,y]))))
                          


    def render(self, beams):
        P = np.empty((len(self.sensor_matrix[0])*len(self.sensor_matrix[1]), 3))
        A2C=np.eye(4)
        A2C = self.pose.pose_matrix.copy()
        for i in range(self.sensor_matrix.shape[0]):
            for j in range(self.sensor_matrix.shape[1]):
                A2C[:3, :3] =self.sensor_matrix[i][j][:3, :3]
                A2C[:3, :3]=np.matmul(A2C[:3, :3],self.R_render[:3, :3])
                beams[self.sensor_matrix.shape[1]*i+j].set_data(P, A2C.copy())
        print(len(beams))
        return beams


    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix


    def init_render_sensor(self, fig):
        R=[[1,0,0],[0,1,0],[0,0,1]]
        P = np.zeros((20, 3))
        colors = np.empty((20-1, 3))
        for d in range(colors.shape[1]):
            P[:, d] = np.linspace(0, self.sensor_range, len(P))
            colors[:, d] = np.linspace(0, 1, len(colors))
        eye=np.eye(4)
        eye[:3,:3]=R
        lines = list()
        for _ in range(self.num_beams*self.num_rays):
            lines.append(fig.plot(P, eye, colors))

        #fig.view_init()

        return fig, lines
    
