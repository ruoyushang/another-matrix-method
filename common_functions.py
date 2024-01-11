
import os, sys
import ROOT
import numpy as np

matrix_rank = 3

min_NImages = 3
max_Roff = 2.0
max_EmissionHeight_cut = 20.
min_EmissionHeight_cut = 6.
max_Rcore = 400.
min_Rcore = 0.

xoff_bins = 20
xoff_start = -2.
xoff_end = 2.
yoff_bins = 20
yoff_start = -2.
yoff_end = 2.
gcut_bins = 4
gcut_start = -0.6
gcut_end = -0.6+1.2*gcut_bins
logE_bins = 6
logE_start = -1.
logE_end = 1.

def ReadRunListFromFile(input_file):

    runlist = []

    inputFile = open(input_file)
    for line in inputFile:
        runlist += [int(line)]

    return runlist

class MyArray3D:

    def __init__(self,x_bins=10,start_x=0.,end_x=10.,y_bins=10,start_y=0.,end_y=10.,z_bins=10,start_z=0.,end_z=10.,overflow=True):
        array_shape = (x_bins,y_bins,z_bins)
        self.delta_x = (end_x-start_x)/float(x_bins)
        self.delta_y = (end_y-start_y)/float(y_bins)
        self.delta_z = (end_z-start_z)/float(z_bins)
        self.xaxis = np.zeros(array_shape[0]+1)
        self.yaxis = np.zeros(array_shape[1]+1)
        self.zaxis = np.zeros(array_shape[2]+1)
        self.waxis = np.zeros(array_shape)
        self.overflow = overflow
        for idx in range(0,len(self.xaxis)):
            self.xaxis[idx] = start_x + idx*self.delta_x
        for idx in range(0,len(self.yaxis)):
            self.yaxis[idx] = start_y + idx*self.delta_y
        for idx in range(0,len(self.zaxis)):
            self.zaxis[idx] = start_z + idx*self.delta_z

    def reset(self):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    self.waxis[idx_x,idx_y,idx_z] = 0.

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]+add_array.waxis[idx_x,idx_y,idx_z]*factor

    def get_bin(self, value_x, value_y, value_z):
        key_idx_x = -1
        key_idx_y = -1
        key_idx_z = -1
        for idx_x in range(0,len(self.xaxis)-1):
            if self.xaxis[idx_x]<=value_x and self.xaxis[idx_x+1]>value_x:
                key_idx_x = idx_x
        for idx_y in range(0,len(self.yaxis)-1):
            if self.yaxis[idx_y]<=value_y and self.yaxis[idx_y+1]>value_y:
                key_idx_y = idx_y
        for idx_z in range(0,len(self.zaxis)-1):
            if self.zaxis[idx_z]<=value_z and self.zaxis[idx_z+1]>value_z:
                key_idx_z = idx_z
        if value_x>self.xaxis.max():
            key_idx_x = len(self.xaxis)-2
        if value_y>self.yaxis.max():
            key_idx_y = len(self.yaxis)-2
        if value_z>self.zaxis.max():
            key_idx_z = len(self.zaxis)-2
        return [key_idx_x,key_idx_y,key_idx_z]

    def fill(self, value_x, value_y, value_z, weight=1.):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x==-1: 
            key_idx_x = 0
            if not self.overflow: weight = 0.
        if key_idx_y==-1: 
            key_idx_y = 0
            if not self.overflow: weight = 0.
        if key_idx_z==-1: 
            key_idx_z = 0
            if not self.overflow: weight = 0.
        if key_idx_x==len(self.xaxis): 
            key_idx_x = len(self.xaxis)-2
            if not self.overflow: weight = 0.
        if key_idx_y==len(self.yaxis): 
            key_idx_y = len(self.yaxis)-2
            if not self.overflow: weight = 0.
        if key_idx_z==len(self.zaxis): 
            key_idx_z = len(self.zaxis)-2
            if not self.overflow: weight = 0.
        self.waxis[key_idx_x,key_idx_y,key_idx_z] += 1.*weight
    
    def divide(self, add_array):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    if add_array.waxis[idx_x,idx_y,idx_z]==0.:
                        self.waxis[idx_x,idx_y,idx_z] = 0.
                    else:
                        self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]/add_array.waxis[idx_x,idx_y,idx_z]

    def get_bin_center(self, idx_x, idx_y, idx_z):
        return [self.xaxis[idx_x]+0.5*self.delta_x,self.yaxis[idx_y]+0.5*self.delta_y,self.zaxis[idx_z]+0.5*self.delta_z]

    def get_bin_content(self, value_x, value_y, value_z):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x==-1: 
            key_idx_x = 0
        if key_idx_y==-1: 
            key_idx_y = 0
        if key_idx_z==-1: 
            key_idx_z = 0
        if key_idx_x==len(self.xaxis): 
            key_idx_x = len(self.xaxis)-2
        if key_idx_y==len(self.yaxis): 
            key_idx_y = len(self.yaxis)-2
        if key_idx_z==len(self.zaxis): 
            key_idx_z = len(self.zaxis)-2
        return self.waxis[key_idx_x,key_idx_y,key_idx_z]

class MyArray1D:

    def __init__(self,x_bins=10,start_x=0.,end_x=10.,overflow=True):
        array_shape = (x_bins)
        self.delta_x = (end_x-start_x)/float(x_bins)
        self.xaxis = np.zeros(array_shape+1)
        self.waxis = np.zeros(array_shape)
        self.overflow = overflow
        for idx in range(0,len(self.xaxis)):
            self.xaxis[idx] = start_x + idx*self.delta_x

    def reset(self):
        for idx_x in range(0,len(self.xaxis)-1):
            self.waxis[idx_x] = 0.

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)-1):
            self.waxis[idx_x] = self.waxis[idx_x]+add_array.waxis[idx_x]*factor

    def get_bin(self, value_x):
        key_idx_x = -1
        for idx_x in range(0,len(self.xaxis)-1):
            if self.xaxis[idx_x]<=value_x and self.xaxis[idx_x+1]>value_x:
                key_idx_x = idx_x
        if value_x>self.xaxis.max():
            key_idx_x = len(self.xaxis)-2
        return key_idx_x

    def fill(self, value_x, weight=1.):
        key_idx = self.get_bin(value_x)
        if key_idx==-1: 
            key_idx = 0
            if not self.overflow: weight = 0.
        if key_idx==len(self.xaxis): 
            key_idx = len(self.xaxis)-2
            if not self.overflow: weight = 0.
        self.waxis[key_idx] += 1.*weight
    
    def divide(self, add_array):
        for idx_x in range(0,len(self.xaxis)-1):
            if add_array.waxis[idx_x]==0.:
                self.waxis[idx_x] = 0.
            else:
                self.waxis[idx_x] = self.waxis[idx_x]/add_array.waxis[idx_x]

    def get_bin_center(self, idx_x):
        return self.xaxis[idx_x]+0.5*self.delta_x

    def get_bin_content(self, value_x):
        key_idx = self.get_bin(value_x)
        if key_idx==-1: 
            key_idx = 0
        if key_idx==len(self.xaxis): 
            key_idx = len(self.xaxis)-2
        return self.waxis[key_idx]

def build_big_camera_matrix(smi_input,runlist):

    big_matrix = []
    for logE in range(0,logE_bins):
        big_matrix += [None]

    logE_axis = MyArray1D(x_bins=logE_bins,start_x=logE_start,end_x=logE_end)

    for run_number in runlist:
    
        rootfile_name = f'{smi_input}/{run_number}.anasum.root'
        print (rootfile_name)
        if not os.path.exists(rootfile_name):
            print (f'file does not exist.')
            continue
    
        xyoff_map = []
        for logE in range(0,logE_bins):
            xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    
        InputFile = ROOT.TFile(rootfile_name)
        TreeName = f'run_{run_number}/stereo/DL3EventTree'
        EvtTree = InputFile.Get(TreeName)
        total_entries = EvtTree.GetEntries()
        print (f'total_entries = {total_entries}')
        for entry in range(0,total_entries):
            EvtTree.GetEntry(entry)
            Xoff = EvtTree.Xoff
            Yoff = EvtTree.Yoff
            MSCW = EvtTree.MSCW
            MSCL = EvtTree.MSCL
            Energy = EvtTree.Energy
            NImages = EvtTree.NImages
            EmissionHeight = EvtTree.EmissionHeight
            Xcore = EvtTree.XCore
            Ycore = EvtTree.YCore
            Roff = pow(Xoff*Xoff+Yoff*Yoff,0.5)
            Rcore = pow(Xcore*Xcore+Ycore*Ycore,0.5)
            logE_bin = logE_axis.get_bin(np.log10(Energy))
            if NImages<min_NImages: continue
            if EmissionHeight>max_EmissionHeight_cut: continue
            if EmissionHeight<min_EmissionHeight_cut: continue
            if Roff>max_Roff: continue
            xyoff_map[logE_bin].fill(Xoff,Yoff,MSCW)
    
        for logE in range(0,logE_bins):
            xyoff_map_1d = []
            for gcut in range(0,gcut_bins):
                for idx_x in range(0,xoff_bins):
                    for idx_y in range(0,yoff_bins):
                        xyoff_map_1d += [xyoff_map[logE].waxis[idx_x,idx_y,gcut]]
            if big_matrix[logE]==None:
                big_matrix[logE] = [xyoff_map_1d]
            else:
                big_matrix[logE] += [xyoff_map_1d]

    return big_matrix

