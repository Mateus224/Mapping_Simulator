import Env.pytransform3d_.visualizer as pv
from Env.pytransform3d_.rotations import *
from Env.pytransform3d_.transformations import *
from Env.pytransform3d_.batch_rotations import *
import open3d as o3d
from Env.pytransform3d_.plot_utils import plot_box
from Agents.lawnMower import lawnMower
from Env.load_env import Load_env
from Env.env3d import  Env
from Agents.DDDQN.DDDQN_agent import DDDQN_agent
import xxhash
from random import randrange
import sys
import os
import copy

obs = None
sumreward = 0



hash_box_map={}
def update_map(fig, step, belief, update_map):
    for i in range(len(update_map)):
        hashkey=update_map[i]
        if hashkey!=0:
            #print(hashkey)
            box=hash_box_map.get(hashkey)
            #if step==0:
            #    box2=copy.deepcopy(box)
            
            
            x= int(hashkey / 1000000)
            y= int((hashkey - (x*1000000))/1000 )
            z= int(hashkey-(x*1000000)-(y*1000))
            value=belief[x,y]
            box.update_color2(fig,[1-value,1-value,0.5])
            #if step==0:
            #    box.remove_artist(fig)
                #vox = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
                #box = pv.Box(size=[1,1,1],A2B=vox, c=[1,0.89,0.707])
            #    box2.add_artist(fig)

        



def animation_callback1(step, n_frames, frame, frame_debug, uav, uuv, uav_beams, uuv_beams, env, agent, fig, b_multiagent):
    global obs
    global sumreward
    reward=0

    if True:
        action = agent.make_action(obs, True)
        obs, reward, done, _, _ , _= env.step(action, all_actions=True, h_level=False, agent="rainbow")
        uav_pose=env.agentDispatcher.uav.pose.pose_matrix.copy()
        if b_multiagent:
            uuv_pose=env.agentDispatcher.uuv.pose.pose_matrix.copy()

        ##--uav stl model has to be centered--##
        if False:
            uuv_pose[0,3]=uav_pose[0,3]-2.1
            uuv_pose[1,3]=uav_pose[1,3]+3.55
            uuv_pose[1,3]=uav_pose[1,3]-1.2

        #print(uav_pose,"...", uuv_pose)
        #agent.store_entropy(entropy,env.t)
        sumreward=sumreward+reward
        belief=env.belief
        update_map_=env.agentDispatcher.update_map
        uav_beams, uuv_beams=env.agentDispatcher.render_sensors(uav_beams, uuv_beams)
        
        uav.set_data(uav_pose)
        if b_multiagent:
            uuv.set_data(uuv_pose)
        if(step>0):
            #first step ignor because of a visual bug
            update_map(fig, step, belief, update_map_)
    else:
        agent._plot_line()

    if b_multiagent:
        return uav, uuv, uav_beams, uuv_beams
    else:
        return uav, uav_beams





def build_env(fig, env, surface_water):
    global boxes
    #voxel = env.load_VM()
    voxel=env
    for xInd, X in enumerate(voxel):
        for yInd, Y in enumerate(X):
            for zInd, Z in enumerate(Y):
                if(Z==1 or Z==0.5):
                    hashkey = 1000000*xInd+1000*yInd+zInd
                    vox = np.array([[1, 0, 0, xInd], [0, 1, 0, yInd], [0, 0, 1, zInd], [0, 0, 0, 1]])
                    if zInd>surface_water:
                        box = pv.Box(size=[1,1,1],A2B=vox, c=[1,1,0.9])
                    elif zInd==surface_water:
                        box = pv.Box(size=[1,1,1],A2B=vox, c=[0.9,0.9,0.9])
                    else:
                        box = pv.Box(size=[1,1,1],A2B=vox, c=[1,0.89,0.707])
                    box.add_artist2(fig)
                    hash_box_map[hashkey] = box
                    if Z==0.5:
                        box2 = pv.Box(size=[1,1,1],A2B=vox, c=[.5,0.5,.5])
                        box2.add_artist(fig)
                    else:
                        box.add_artist(fig)
                if((xInd  == 0) and (yInd  == 0)):
                    wall = np.array([[1, 0, 0, -1], [0, 1, 0,( voxel.shape[1]/2)+.5], [0, 0, 1, (voxel.shape[2]/4)+.5], [0, 0, 0, 1]])
                    box = pv.Box(size=[1, voxel.shape[1], voxel.shape[2]/2], A2B=wall)
                    box.add_artist2(fig)
                    box.add_artist(fig)

                    wall = np.array([ [1, 0, 0, (voxel.shape[1] / 2) +.5], [0, 1, 0, -1],[0, 0, 1, (voxel.shape[2]/4) + .5],[0, 0, 0, 1]])
                    box = pv.Box(size=[ voxel.shape[0], 1,voxel.shape[2]/2], A2B=wall)
                    box.add_artist(fig)
                    box.add_artist2(fig)
                if ((xInd == 0) and (yInd == 0)and (zInd == 0)):
                    wall = np.array(
                        [[1, 0, 0, (voxel.shape[1] / 2) +.5], [0, 1, 0, (voxel.shape[1] / 2) +.5], [0, 0, 1, -1],
                         [0, 0, 0, 1]])
                    box = pv.Box(size=[voxel.shape[0], voxel.shape[1], 1], A2B=wall)
                    box.add_artist(fig)
                    box.add_artist2(fig)

    return fig

def init_env(surface_water, b_multiagent):

    BASE_DIR = "Mashes/"
    data_dir = BASE_DIR
    search_path = "."
    while (not os.path.exists(data_dir) and
           os.path.dirname(search_path) != "pytransform3d_"):
        search_path = os.path.join(search_path, "..")
        data_dir = os.path.join(search_path, BASE_DIR)
    fig = pv.figure(width=500, height=500)
    frame = fig.plot_basis(R=np.eye(3), s=2)
    frame_debug = fig.plot_basis(R=np.eye(3), s=2)
    R = matrix_from_angle(2,3*np.pi/2)
    A2C = np.eye(4)
    A2C[:3, :3] = R
    if b_multiagent:
        uuv = pv.Mesh("Mashes/uav.stl",s=[0.5,0.5,0.5], c=[0.9,0.1,0.1])
        uuv.add_artist2(fig)
        uuv.add_artist(fig)
    else:
        uuv=None
    uav = pv.Mesh("Mashes/uuv.stl",s=[0.3,0.225,0.3], c=[0.4,0.3,0.2])#c=[0.0,0.7,0.9])
    water = pv.Mesh("Mashes/water.stl",s=[0.005,0.005,0.005], c=[0.,0.,0.2])
    uav.add_artist2(fig)
    uav.add_artist(fig)
    water.add_artist2(fig)
    water.add_artist(fig)
    R= matrix_from_axis_angle([1,0,0,np.pi/2])
    water_pose= transform_from(R,[13,13,surface_water])
    water.set_data(water_pose)

    return fig, uuv, uav, frame, frame_debug

def init_render(args, agent, env, config):

    global obs
    b_multiagent=config.getboolean('ENV','mutiagent')
    obs, voxelVis = env.reset(h_level=False, validation=True)
    surface_water=env.loaded_env._2_D_min+25
    fig, uuv, uav, frame, frame_debug = init_env(surface_water, b_multiagent)
    fig = build_env(fig, voxelVis, surface_water)
    fig, uav_beams, uuv_beams = env.agentDispatcher.init_render_sensors(fig, b_multiagent)
    n_frames = 900
    fig.animate(animation_callback1, n_frames, fargs=(n_frames, frame, frame_debug, uav, uuv, uav_beams, uuv_beams, env, agent, fig, b_multiagent), loop=True)
    fig.show()


