import numpy as np
from pyntcloud import PyntCloud
import csv
from numpy.random import default_rng
import random


class Load_env():
    """"This class loads and preprocess the
    simulated environment"""
    def __init__(self, env_shape):
        self.xn=env_shape[0]
        self.yn=env_shape[1]
        self.zn=env_shape[2]
        self.hashmap = {}
        self.map_2_5D= np.zeros((self.xn, self.yn, 2))
        self.litter_amount=63
        self.litter_perc=0.1
        self._2_D_min=999


    def store_as_Hash_map(self, voxel):
        """Store the voxelmap in a hashmap
        which can handle up to 1000*1000*1000 voxels
        """
        for xInd, X in enumerate(voxel):
            for yInd, Y in enumerate(X):
                z_litter=False
                for zInd, Z in enumerate(Y):
                    if (Z == 1 or Z == 0.5):
                        hashkey = 1000000*xInd+1000*yInd+zInd
                        self.hashmap[hashkey] = Z
                        self.map_2_5D[xInd,yInd,0]=zInd
                        if(Z==0.5):
                            z_litter=True
                            self.map_2_5D[xInd,yInd,1]=1
                        elif(Z==1):
                            if(z_litter):
                                self.map_2_5D[xInd,yInd,1]=0
        #self.map_2_5D=self.map_2_5D*4
        #img = Image.fromarray(self.map_2_5D)
        #img.show()




    def load_VM(self, pc='Env/xyz_env/sample1.xyz', upSideDown=True):#output.xyz point_cloud
        #random_map=1
        #cloud=PyntCloud(pd.DataFrame(Cloud, columns=["x","y","z"]))

        cloud = PyntCloud.from_file(pc,
                                   sep=" ",
                                   header=0,
                                  names=["x","y","z"])

        rot=np.random.random_integers(8)-1
        #rot=7
        upSideDown_rand=False
        #upSideDown=True
        if upSideDown:
            upSideDown_rand=np.random.random_integers(2)-1

        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=self.xn, n_y=self.yn, n_z=self.zn)
        voxelgrid = cloud.structures[voxelgrid_id]
        if rot==0:
            x_cords = -voxelgrid.voxel_x
            y_cords = -voxelgrid.voxel_y
        elif rot==1:
            y_cords = -voxelgrid.voxel_x
            x_cords = -voxelgrid.voxel_y
        elif rot==2:
            y_cords = -voxelgrid.voxel_y
            x_cords = voxelgrid.voxel_x
        elif rot==3:
            y_cords = voxelgrid.voxel_y
            x_cords = -voxelgrid.voxel_x
        elif rot==4:
            y_cords = -voxelgrid.voxel_x
            x_cords = voxelgrid.voxel_y
        elif rot==5:
            y_cords = voxelgrid.voxel_x
            x_cords = -voxelgrid.voxel_y
        elif rot==6:
            y_cords = voxelgrid.voxel_y
            x_cords = voxelgrid.voxel_x
        elif rot==7:
            y_cords = voxelgrid.voxel_x
            x_cords = voxelgrid.voxel_y
        if upSideDown_rand:
            z_cords = -voxelgrid.voxel_z
        else:
            z_cords = voxelgrid.voxel_z
        voxel = np.zeros((self.xn, self.yn, self.zn))

        for x, y, z in zip(x_cords, y_cords, z_cords):
            voxel[x][y][z] = 1
            
        minimum= self.search_minimum(voxel)
        voxel= self.shift_to_minimum(voxel, minimum)
        _2_D_min= self.find_2_D_minimum(voxel)
        self._2_D_min=_2_D_min
        layer0, layer1, layer2, layer3, layer4, layer5, layer6 = self.store_layers(voxel, _2_D_min)
        self.voxel_map= self.place_litter(voxel, layer0, layer1, layer2, layer3, layer4, layer5, layer6)
        self.store_as_Hash_map(voxel)

        return self.voxel_map, self.hashmap

    def search_minimum(self, voxel):
        tmp=10000
        for xInd, X in enumerate(voxel):
            for yInd, Y in enumerate(X):
                for zInd, Z in enumerate(Y):
                    if(Z==1):
                        if(zInd<tmp):
                            tmp=zInd
        return tmp


    def shift_to_minimum(self, voxel, minimum):
        #env_2D_min=search2_5_minimum()
        for xInd, X in enumerate(voxel):
            for yInd, Y in enumerate(X):
                tmp=False
                for zInd, Z in enumerate(Y):
                    if(Z==1):
                        voxel[xInd, yInd, zInd]=0
                        new_zInd=zInd-minimum
                        voxel[xInd, yInd, new_zInd]=1
        return voxel

    def find_2_D_minimum(self, voxel):
        _2_D_minimum= 99999
        for xInd, X in enumerate(voxel):
            for yInd, Y in enumerate(X):
                for zInd, Z in enumerate(Y):
                    if(Z==1):
                        if(voxel[xInd][yInd][zInd+1]==False and voxel[xInd][yInd][zInd+2]==False):
                            if zInd<_2_D_minimum:
                                _2_D_minimum=zInd

        return _2_D_minimum


    def store_layers(self, voxel, _2_D_min):
        
        tmp_zInd=0
        layer0=[]
        layer1=[]
        layer2=[]
        layer3=[]
        layer4=[]
        layer5=[]
        layer6=[]
        #rng = rng.standard_normal()
        vox_swapped=np.swapaxes(voxel,0,2)
        for zInd, Z in enumerate(vox_swapped):
            for yInd, Y in enumerate(Z):
                for xInd, X in enumerate(Y):
                    if(X==1):
                        if(voxel[xInd][yInd][zInd+1]==False):
                            if (zInd)==_2_D_min:
                                layer0.append([xInd,yInd,zInd])
                            elif (zInd)==_2_D_min+1:
                                layer1.append([xInd,yInd,zInd])
                            elif (zInd)==_2_D_min+2:
                                layer2.append([xInd,yInd,zInd])
                            elif (zInd)==_2_D_min+3:
                                layer3.append([xInd,yInd,zInd])
                            elif (zInd)==_2_D_min+4:
                                layer4.append([xInd,yInd,zInd])
                            elif (zInd)==_2_D_min+5:
                                layer5.append([xInd,yInd,zInd])
                            elif (zInd)==_2_D_min+6:
                                layer6.append([xInd,yInd,zInd])
                            #rints = rng.integers(low=0, high=100, size=1)
                            #if (zInd)>=_2_D_min:
                               # if (zInd*zInd)<=(rints+_2_D_min):
                                    #if self.litter_amount<= (self.xn*self.yn*(self.litter_perc)): 
                                     #   self.litter_amount=self.litter_amount+1
                            #voxel[xInd][yInd][zInd] = 0.5
        return layer0, layer1, layer2, layer3, layer4, layer5, layer6

    def place_litter(self, voxel,layer0, layer1, layer2, layer3, layer4, layer5,layer6):
        rng_seed= random.randint(2,99999)
        #rng_seed=1234
        rng = default_rng(rng_seed)
        """
        for xInd,yInd,zInd in layer0:
            voxel[xInd][yInd][zInd] = 0.5
        for xInd,yInd,zInd in layer1:
            voxel[xInd][yInd][zInd] = 0.5
        for xInd,yInd,zInd in layer2:
            voxel[xInd][yInd][zInd] = 0.5
        #for xInd,yInd,zInd in layer2:
        #    voxel[xInd][yInd][zInd] = 0.5
        return voxel
        """
       
            
        if len(layer0)*0.8<= self.litter_amount: 
            for xInd,yInd,zInd in layer0:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<self.litter_amount and self.litter_amount>0: #40%
                    self.litter_amount=self.litter_amount-1
                    voxel[xInd][yInd][zInd] = 0.5
        else:
            prop=int((self.litter_amount/len(layer0))*100)+1
            for xInd,yInd,zInd in layer0:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<prop: #40%
                    if self.litter_amount>0:
                        self.litter_amount=self.litter_amount-1
                        voxel[xInd][yInd][zInd] = 0.5

        if len(layer1)*0.6<= self.litter_amount:
            for xInd,yInd,zInd in layer1:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<self.litter_amount and self.litter_amount>0: #40%
                    self.litter_amount=self.litter_amount-1
                    voxel[xInd][yInd][zInd] = 0.5
        else:
            prop=int((self.litter_amount/len(layer1))*100)+1
            for xInd,yInd,zInd in layer1:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<prop: #40%
                    if self.litter_amount>0:
                        self.litter_amount=self.litter_amount-1
                        voxel[xInd][yInd][zInd] = 0.5
        if len(layer2)*0.43<= self.litter_amount:
            for xInd,yInd,zInd in layer2:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<self.litter_amount and self.litter_amount>0: #40%
                    self.litter_amount=self.litter_amount-1
                    voxel[xInd][yInd][zInd] = 0.5
        else:
            prop=int((self.litter_amount/len(layer2))*100)+1
            for xInd,yInd,zInd in layer2:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<prop: #40%
                    if self.litter_amount>0:
                        self.litter_amount=self.litter_amount-1
                        voxel[xInd][yInd][zInd] = 0.5
        if len(layer3)*0.34<= self.litter_amount:
            for xInd,yInd,zInd in layer3:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<self.litter_amount and self.litter_amount>0: #40%
                    self.litter_amount=self.litter_amount-1
                    voxel[xInd][yInd][zInd] = 0.5
        else:
            prop=int((self.litter_amount/len(layer3))*100)+1
            for xInd,yInd,zInd in layer3:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<prop: #40%
                    if self.litter_amount>0:
                        self.litter_amount=self.litter_amount-1
                        voxel[xInd][yInd][zInd] = 0.5
        if len(layer4)*0.21<= self.litter_amount:
            for xInd,yInd,zInd in layer4:
                rints = rng.integers(low=0, high=100, size=1) 
                if rints<self.litter_amount: #40%
                    self.litter_amount=self.litter_amount-1
                    voxel[xInd][yInd][zInd] = 0.5
        else:
            prop=int((self.litter_amount/len(layer4))*100)+1
            for xInd,yInd,zInd in layer4:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<prop: #40%
                    if self.litter_amount>0:
                        self.litter_amount=self.litter_amount-1
                        voxel[xInd][yInd][zInd] = 0.5
        if len(layer5)*0.14<= self.litter_amount:
            for xInd,yInd,zInd in layer5:
                rints = rng.integers(low=0, high=100, size=1) 
                if rints<self.litter_amount: #40%
                    self.litter_amount=self.litter_amount-1
                    voxel[xInd][yInd][zInd] = 0.5
        else:
            prop=int((self.litter_amount/len(layer5))*100)+1
            for xInd,yInd,zInd in layer5:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<prop: #40%
                    if self.litter_amount>0:
                        self.litter_amount=self.litter_amount-1
                        voxel[xInd][yInd][zInd] = 0.5
        if len(layer6)*0.1<= self.litter_amount:
            for xInd,yInd,zInd in layer6:
                rints = rng.integers(low=0, high=100, size=1) 
                if rints<self.litter_amount: #40%
                    self.litter_amount=self.litter_amount-1
                    voxel[xInd][yInd][zInd] = 0.5
        else:
            prop=int((self.litter_amount/len(layer6))*100)+1
            for xInd,yInd,zInd in layer6:
                rints = rng.integers(low=0, high=100, size=1)
                if rints<prop: #40%
                    if self.litter_amount>0:
                        self.litter_amount=self.litter_amount-1
                        voxel[xInd][yInd][zInd] = 0.5
        
        return voxel
        
        
    def readData(self):
        with open('xyz_env/1.xyz') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            next(csv_reader, None)
            data = [data for data in csv_reader]
            data_array = np.asarray(data, dtype=np.float64)
            data_array = np.delete(data_array, 4, axis=1)
            data_array = np.delete(data_array, 3, axis=1)
        return data_array


if __name__ == "__main__":
        pointCloud='xyz_env/1.xyz'
        PCL=readData()
        load_VM(PCL)
