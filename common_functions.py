
import os, sys
import ROOT
import numpy as np
import pickle
import csv
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
import tracemalloc

min_NImages = 3
max_Roff = 2.0
max_EmissionHeight_cut = 20.
min_EmissionHeight_cut = 6.
max_Rcore = 250.
min_Rcore = 50.
min_Energy_cut = 0.2
max_Energy_cut = 10.0
MSCW_cut = 0.6
MSCL_cut = 0.6
MVA_cut = 0.5

xoff_start = -2.
xoff_end = 2.
yoff_start = -2.
yoff_end = 2.
gcut_bins = 3
gcut_start = 0
gcut_end = gcut_bins

logE_bins = [-0.75,-0.625,-0.50,-0.375,-0.25,0.00,0.25,0.50,0.75,1.0] # logE TeV
#logE_bins = [-0.625,-0.50,-0.375,-0.25,0.00,0.25,0.50,0.75,1.0] # logE TeV
logE_nbins = len(logE_bins)-1

skymap_size = 3.
skymap_bins = 30

doFluxCalibration = True
#doFluxCalibration = False
calibration_radius = 0.15 # need to be larger than the PSF and smaller than the integration radius

logE_min = 0
logE_max = logE_nbins
matrix_rank = 0.1
#xoff_bins = [10,5,4,4,3,3,1,1]
#xoff_bins = [10,10,10,10,3,3,1,1]
xoff_bins = [7,7,7,7,7,7,7,7,7]
yoff_bins = xoff_bins

chi2_cut = 0.5
#chi2_cut = 1e10

smi_aux = os.environ.get("SMI_AUX")
smi_dir = os.environ.get("SMI_DIR")

def ReadRunListFromFile(input_on_file,input_off_file):

    on_runlist = []
    off_runlist = []

    on_inputFile = open(input_on_file)
    for on_line in on_inputFile:
        on_runlist += [int(on_line)]
        off_inputFile = open(input_off_file)
        paired_off_runs = []
        for off_line in off_inputFile:
            line_split = off_line.split()
            on_run = int(line_split[0])
            off_run = int(line_split[1])
            if on_run==int(on_line):
                paired_off_runs += [off_run]
        off_runlist += [paired_off_runs]

    return on_runlist, off_runlist

def ReadOffRunListFromFile(input_file):

    runlist = []

    inputFile = open(input_file)
    for line in inputFile:
        line_split = line.split()
        runlist += [int(line_split[1])]

    return runlist

def smooth_image(image_data,xaxis,yaxis,kernel_radius=0.07):

    image_smooth = np.zeros_like(image_data)

    bin_size = abs(xaxis[1]-xaxis[0])

    kernel_pix_size = int(kernel_radius/bin_size)
    if kernel_pix_size==0: return

    gaus_norm = 2.*np.pi*kernel_radius*kernel_radius
    image_kernel = np.zeros_like(image_data)
    central_bin_x = int(float(len(xaxis)-1)/2.)
    central_bin_y = int(float(len(yaxis)-1)/2.)
    for idx_x in range(0,len(xaxis)-1):
        for idx_y in range(0,len(yaxis)-1):
            pix_x = xaxis[idx_x] - xaxis[central_bin_x]
            pix_y = yaxis[idx_y] - yaxis[central_bin_y]
            distance = pow(pix_x*pix_x+pix_y*pix_y,0.5)
            pix_content = np.exp(-(distance*distance)/(2.*kernel_radius*kernel_radius))
            image_kernel[idx_y,idx_x] = pix_content
            #image_kernel[idx_y,idx_x] = pix_content/gaus_norm

    kernel_norm = np.sum(image_kernel)
    for idx_x1 in range(0,len(xaxis)-1):
        for idx_y1 in range(0,len(yaxis)-1):
            image_smooth[idx_y1,idx_x1] = 0.
            for idx_x2 in range(idx_x1-2*kernel_pix_size,idx_x1+2*kernel_pix_size):
                for idx_y2 in range(idx_y1-2*kernel_pix_size,idx_y1+2*kernel_pix_size):
                    if idx_x2<0: continue
                    if idx_y2<0: continue
                    if idx_x2>=len(xaxis)-1: continue
                    if idx_y2>=len(yaxis)-1: continue
                    old_content = image_data[idx_y2,idx_x2]
                    scale = image_kernel[central_bin_y+idx_y2-idx_y1,central_bin_x+idx_x2-idx_x1]
                    image_smooth[idx_y1,idx_x1] += old_content*scale
                    #image_smooth[idx_y1,idx_x1] += old_content*scale/kernel_norm
                    #image_smooth[idx_y1,idx_x1] += old_content

    for idx_x in range(0,len(xaxis)-1):
        for idx_y in range(0,len(yaxis)-1):
            image_data[idx_y,idx_x] = image_smooth[idx_y,idx_x]

class MyArray3D:

    def __init__(self,x_bins=10,start_x=0.,end_x=10.,y_bins=10,start_y=0.,end_y=10.,z_bins=10,start_z=0.,end_z=10.,overflow=False):
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

    def scale(self, factor):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]*factor

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]+add_array.waxis[idx_x,idx_y,idx_z]*factor

    def addSquare(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    self.waxis[idx_x,idx_y,idx_z] = pow(pow(self.waxis[idx_x,idx_y,idx_z],2)+pow(add_array.waxis[idx_x,idx_y,idx_z]*factor,2),0.5)

    def get_bin(self, value_x, value_y, value_z):
        key_idx_x = -1
        key_idx_y = -1
        key_idx_z = -1
        for idx_x in range(0,len(self.xaxis)-1):
            if abs(self.xaxis[idx_x]-value_x)<=abs(self.delta_x) and abs(self.xaxis[idx_x+1]-value_x)<abs(self.delta_x):
                key_idx_x = idx_x
                break
        for idx_y in range(0,len(self.yaxis)-1):
            if abs(self.yaxis[idx_y]-value_y)<=abs(self.delta_y) and abs(self.yaxis[idx_y+1]-value_y)<abs(self.delta_y):
                key_idx_y = idx_y
                break
        for idx_z in range(0,len(self.zaxis)-1):
            if abs(self.zaxis[idx_z]-value_z)<=abs(self.delta_z) and abs(self.zaxis[idx_z+1]-value_z)<abs(self.delta_z):
                key_idx_z = idx_z
                break
        return [key_idx_x,key_idx_y,key_idx_z]

    def fill(self, value_x, value_y, value_z, weight=1.):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x==-1: 
            key_idx_x = 0
            weight = 0.
        if key_idx_y==-1: 
            key_idx_y = 0
            weight = 0.
        if key_idx_z==-1: 
            key_idx_z = 0
            weight = 0.
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
            return 0.
        if key_idx_y==-1: 
            return 0.
        if key_idx_z==-1: 
            return 0.
        #if key_idx_x==len(self.xaxis): 
        #    key_idx_x = len(self.xaxis)-2
        #if key_idx_y==len(self.yaxis): 
        #    key_idx_y = len(self.yaxis)-2
        #if key_idx_z==len(self.zaxis): 
        #    key_idx_z = len(self.zaxis)-2
        return self.waxis[key_idx_x,key_idx_y,key_idx_z]

class MyArray1D:

    def __init__(self,x_nbins=10,start_x=0.,end_x=10.,x_bins=[],overflow=False):
        if len(x_bins)==0:
            array_shape = (x_nbins)
            self.delta_x = np.empty(x_nbins+1)
            self.delta_x.fill((end_x-start_x)/float(x_nbins))
            self.xaxis = np.zeros(array_shape+1)
            self.waxis = np.zeros(array_shape)
            self.overflow = overflow
            for idx in range(0,len(self.xaxis)):
                self.xaxis[idx] = start_x + idx*self.delta_x[idx]
        else:
            self.xaxis = np.array(x_bins)
            self.waxis = np.zeros(len(x_bins)-1)
            self.overflow = overflow
            self.delta_x = np.empty(len(x_bins))
            for idx in range(0,len(self.xaxis)-1):
                self.delta_x[idx] = self.xaxis[idx+1] - self.xaxis[idx]
            self.delta_x[len(self.xaxis)-1] = self.delta_x[len(self.xaxis)-2]

    def reset(self):
        for idx_x in range(0,len(self.xaxis)-1):
            self.waxis[idx_x] = 0.

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)-1):
            self.waxis[idx_x] = self.waxis[idx_x]+add_array.waxis[idx_x]*factor

    def get_bin(self, value_x):
        key_idx_x = -1
        for idx_x in range(0,len(self.xaxis)):
            if (value_x-self.xaxis[idx_x])<self.delta_x[idx_x]:
                key_idx_x = idx_x
                break
        if value_x>self.xaxis.max():
            key_idx_x = -1
        return key_idx_x

    def fill(self, value_x, weight=1.):
        key_idx = self.get_bin(value_x)
        if key_idx==-1: 
            key_idx = 0
            weight = 0.
        self.waxis[key_idx] += 1.*weight
    
    def divide(self, add_array):
        for idx_x in range(0,len(self.xaxis)-1):
            if add_array.waxis[idx_x]==0.:
                self.waxis[idx_x] = 0.
            else:
                self.waxis[idx_x] = self.waxis[idx_x]/add_array.waxis[idx_x]

    def get_bin_center(self, idx_x):
        return self.xaxis[idx_x]+0.5*self.delta_x[idx_x]

    def get_bin_content(self, value_x):
        key_idx = self.get_bin(value_x)
        if key_idx==-1: 
            key_idx = 0
        if key_idx==len(self.xaxis): 
            key_idx = len(self.xaxis)-2
        return self.waxis[key_idx]

logE_axis = MyArray1D(x_bins=logE_bins)

def GetGammaSources(tele_point_ra, tele_point_dec):

    bright_stars_coord = []
    inputFile = open(f'{smi_aux}/TeVCat_RaDec.txt')
    for line in inputFile:
        if line=='': continue
        line_split = line.split()
        if len(line_split)!=2: continue
        star_ra = float(line_split[0])
        star_dec = float(line_split[1])
        distance = pow(pow(star_ra-tele_point_ra,2)+pow(star_dec-tele_point_dec,2),0.5)
        if distance>2.: continue
        #print (f'{line_split}')
        bright_stars_coord += [[star_ra,star_dec]]
    print (f'Found {len(bright_stars_coord)} Gamma-ray sources.')
    return bright_stars_coord

def GetBrightStars(tele_point_ra, tele_point_dec):

    brightness_cut = 6.0
    bright_stars_coord = []
    inputFile = open(f'{smi_aux}/Hipparcos_MAG8_1997.dat')
    for line in inputFile:
        if '#' in line: continue 
        if '*' in line: continue 
        if line=='': continue
        line_split = line.split()
        if len(line_split)!=5: continue
        star_ra = float(line_split[0])
        star_dec = float(line_split[1])
        star_brightness = float(line_split[3]) + float(line_split[4])
        distance = pow(pow(star_ra-tele_point_ra,2)+pow(star_dec-tele_point_dec,2),0.5)
        if distance>2.: continue
        #print (f'{line_split}')
        if star_brightness<brightness_cut:
            bright_stars_coord += [[star_ra,star_dec]]

    print (f'Found {len(bright_stars_coord)} bright stars.')
    return bright_stars_coord


def CoincideWithBrightStars(ra, dec, bright_stars_coord):
    bright_star_radius_cut = 0.25
    isCoincident = False
    for star in range(0,len(bright_stars_coord)):
        star_ra = bright_stars_coord[star][0]
        star_dec = bright_stars_coord[star][1]
        distance = pow(pow(star_ra-ra,2)+pow(star_dec-dec,2),0.5)
        if distance>bright_star_radius_cut: continue
        isCoincident = True
    return isCoincident

def EventGammaCut(MSCL,MSCW):

    GammaCut = 1e10

    if abs(MSCL)<1. and abs(MSCW)<1.:
        GammaCut = 0.5
    elif abs(MSCL)<1. and abs(MSCW)<2.:
        GammaCut = 1.5
    elif abs(MSCL)<2. and abs(MSCW)<1.:
        GammaCut = 2.5
    #elif abs(MSCL)<2. and abs(MSCW)<2.:
    #    GammaCut = 3.5

    return GammaCut

def build_big_camera_matrix(smi_input,runlist,max_runs=1e10,is_on=True,specific_run=0,control_region=False):

    #big_matrix = []
    #for logE in range(0,logE_nbins):
    #    big_matrix += [None]

    big_matrix = []

    run_count = 0
    for run_number in runlist:

        print (f'{run_count}/{len(runlist)} runs saved.')
    
        rootfile_name = f'{smi_input}/{run_number}.anasum.root'
        print (rootfile_name)
        if not os.path.exists(rootfile_name):
            print (f'file does not exist.')
            continue
        if specific_run!=0:
            if specific_run!=run_number: 
                continue
        run_count += 1
    
        xyoff_map = []
        for logE in range(0,logE_nbins):
            xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    
        InputFile = ROOT.TFile(rootfile_name)

        TreeName = f'run_{run_number}/stereo/pointingDataReduced'
        TelTree = InputFile.Get(TreeName)
        TelTree.GetEntry(int(float(TelTree.GetEntries())/2.))
        TelRAJ2000 = TelTree.TelRAJ2000*180./np.pi
        TelDecJ2000 = TelTree.TelDecJ2000*180./np.pi
        bright_star_coord = GetBrightStars(TelRAJ2000,TelDecJ2000)
        gamma_source_coord = GetGammaSources(TelRAJ2000,TelDecJ2000)

        TreeName = f'run_{run_number}/stereo/DL3EventTree'
        EvtTree = InputFile.Get(TreeName)
        total_entries = EvtTree.GetEntries()
        print (f'total_entries = {total_entries}')
        for entry in range(0,total_entries):
            EvtTree.GetEntry(entry)
            Xoff = EvtTree.Xoff
            Yoff = EvtTree.Yoff
            Xderot = EvtTree.Xderot
            Yderot = EvtTree.Yderot
            MSCW = EvtTree.MSCW/MSCW_cut
            MSCL = EvtTree.MSCL/MSCL_cut
            GammaCut = EventGammaCut(MSCL,MSCW)
            Energy = EvtTree.Energy
            NImages = EvtTree.NImages
            EmissionHeight = EvtTree.EmissionHeight
            Xcore = EvtTree.XCore
            Ycore = EvtTree.YCore
            Roff = pow(Xoff*Xoff+Yoff*Yoff,0.5)
            Rcore = pow(Xcore*Xcore+Ycore*Ycore,0.5)
            logE = logE_axis.get_bin(np.log10(Energy))
            if logE<0: continue
            if logE>=len(xyoff_map): continue
            if NImages<min_NImages: continue
            if EmissionHeight>max_EmissionHeight_cut: continue
            if EmissionHeight<min_EmissionHeight_cut: continue
            if Roff>max_Roff: continue
            if Rcore>max_Rcore: continue
            if Rcore<min_Rcore: continue
            if Energy<min_Energy_cut: continue
            if Energy>max_Energy_cut: continue
            #if not control_region:
            #    if MSCL>1.0: continue
            #else:
            #    if MSCL<1.0: continue
            #    if MSCL>2.0: continue
            if GammaCut>float(gcut_end): continue

            Xsky = TelRAJ2000 + Xderot
            Ysky = TelDecJ2000 + Yderot
            #mirror_Xsky = TelRAJ2000 - Xderot
            #mirror_Ysky = TelDecJ2000 + Yderot
            #found_bright_star = CoincideWithBrightStars(Xsky, Ysky, bright_star_coord)
            #found_gamma_source = CoincideWithBrightStars(Xsky, Ysky, gamma_source_coord)
            #found_mirror_star = CoincideWithBrightStars(mirror_Xsky, mirror_Ysky, bright_star_coord)
            #found_mirror_gamma_source = CoincideWithBrightStars(mirror_Xsky, mirror_Ysky, gamma_source_coord)

            #if not is_on:
            #    if found_bright_star: continue
            #    if found_gamma_source: continue
            #    if found_mirror_star or found_mirror_gamma_source:
            #        xyoff_map[logE].fill(-Xderot,Yderot,GammaCut)

            xyoff_map[logE].fill(Xoff,Yoff,GammaCut)
    
        #for logE in range(0,logE_nbins):
        #    xyoff_map_1d = []
        #    for gcut in range(0,gcut_bins):
        #        for idx_x in range(0,xoff_bins[logE]):
        #            for idx_y in range(0,yoff_bins[logE]):
        #                xyoff_map_1d += [xyoff_map[logE].waxis[idx_x,idx_y,gcut]]
        #    if big_matrix[logE]==None:
        #        big_matrix[logE] = [xyoff_map_1d]
        #    else:
        #        big_matrix[logE] += [xyoff_map_1d]

        xyoff_map_1d = []
        for gcut in range(0,gcut_bins):
            for logE in range(0,logE_nbins):
                for idx_x in range(0,xoff_bins[logE]):
                    for idx_y in range(0,yoff_bins[logE]):
                        xyoff_map_1d += [xyoff_map[logE].waxis[idx_x,idx_y,gcut]]
        big_matrix += [xyoff_map_1d]


        InputFile.Close()
        if run_count==max_runs: break

    return big_matrix

def build_skymap(smi_input,eigenvector_path,runlist,src_ra,src_dec,max_runs=1e10,control_region=False):

    # start memory profiling
    tracemalloc.start()

    xsky_start = src_ra+skymap_size
    xsky_end = src_ra-skymap_size
    ysky_start = src_dec-skymap_size
    ysky_end = src_dec+skymap_size

    print ('loading svd pickle data... ')
    input_filename = eigenvector_path
    big_eigenvectors = pickle.load(open(input_filename, "rb"))

    exposure_hours = 0.
    avg_tel_elev = 0.
    avg_tel_azim = 0.
    data_sky_map = []
    fit_sky_map = []
    for logE in range(0,logE_nbins):
        data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
        fit_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

    data_xyoff_map = []
    fit_xyoff_map = []
    ratio_xyoff_map = []
    for logE in range(0,logE_nbins):
        data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
        fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
        ratio_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

    #effective_matrix_rank = max(1,big_eigenvectors[0].shape[0])
    effective_matrix_rank = max(1,big_eigenvectors.shape[0])
    truth_params = [1e-3] * effective_matrix_rank
    fit_params = [1e-3] * effective_matrix_rank
    cr_qual = 0.
    sr_qual = 0.
    print (f'big_eigenvectors.shape = {big_eigenvectors.shape}') 

    print ('build big matrix...')
    big_on_matrix = build_big_camera_matrix(smi_input,runlist,max_runs=1e10,is_on=True,control_region=control_region)

    #if big_on_matrix[0]==None:
    if len(big_on_matrix)==0:
        print (f'No data. Break.')
        return [exposure_hours,avg_tel_elev,avg_tel_azim,truth_params,fit_params,sr_qual,cr_qual], data_sky_map, fit_sky_map, data_xyoff_map, fit_xyoff_map

    for logE in range(0,logE_nbins):
        data_xyoff_map[logE].reset()
        fit_xyoff_map[logE].reset()
        ratio_xyoff_map[logE].reset()

    print ('fitting xyoff maps...')

    #for logE in range(0,logE_nbins):

    #effective_matrix_rank = max(1,big_eigenvectors[logE].shape[0])
    effective_matrix_rank = max(1,big_eigenvectors.shape[0])

    #data_xyoff_map_1d = np.array(big_on_matrix[logE][0])
    #for entry in range(1,len(big_on_matrix[logE])):
    #    data_xyoff_map_1d += np.array(big_on_matrix[logE][entry])
    data_xyoff_map_1d = np.array(big_on_matrix[0])
    for entry in range(1,len(big_on_matrix)):
        data_xyoff_map_1d += np.array(big_on_matrix[entry])

    print (f'data_xyoff_map_1d.shape = {data_xyoff_map_1d.shape}')
    #truth_params = big_eigenvectors[logE] @ data_xyoff_map_1d
    #truth_xyoff_map_1d = big_eigenvectors[logE].T @ truth_params
    truth_params = big_eigenvectors @ data_xyoff_map_1d
    truth_xyoff_map_1d = big_eigenvectors.T @ truth_params

    init_params = truth_params
    stepsize = [1e-4] * effective_matrix_rank
    solution = minimize(
        cosmic_ray_like_chi2,
        x0=init_params,
        #args=(big_eigenvectors[logE],data_xyoff_map_1d,logE,0),
        args=(big_eigenvectors,data_xyoff_map_1d,0),
        #args=(big_eigenvectors,data_xyoff_map_1d,-1),
        method='L-BFGS-B',
        jac=None,
        options={'eps':stepsize,'ftol':0.0001},
    )
    fit_params = solution['x']
    #fit_xyoff_map_1d = big_eigenvectors[logE].T @ fit_params
    fit_xyoff_map_1d = big_eigenvectors.T @ fit_params

    init_params = truth_params
    stepsize = [1e-4] * effective_matrix_rank
    solution = minimize(
        cosmic_ray_like_chi2,
        x0=init_params,
        #args=(big_eigenvectors[logE],data_xyoff_map_1d,logE,2),
        args=(big_eigenvectors,data_xyoff_map_1d,2),
        method='L-BFGS-B',
        jac=None,
        options={'eps':stepsize,'ftol':0.0001},
    )
    fit_cr_params = solution['x']
    #fit_cr_xyoff_map_1d = big_eigenvectors[logE].T @ fit_cr_params
    fit_cr_xyoff_map_1d = big_eigenvectors.T @ fit_cr_params

    for entry in range(0,len(truth_params)):
        print (f'truth_params = {truth_params[entry]:0.1f}, fit_params = {fit_params[entry]: 0.1f}')

    #run_sr_chi2 = cosmic_ray_like_chi2(fit_params,big_eigenvectors[logE],data_xyoff_map_1d,logE,1)
    #run_cr_chi2 = cosmic_ray_like_chi2(fit_cr_params,big_eigenvectors[logE],data_xyoff_map_1d,logE,3)
    #sr_data_count = cosmic_ray_like_count(data_xyoff_map_1d,logE,region_type=1)
    #fit_data_count = cosmic_ray_like_count(fit_xyoff_map_1d,logE,region_type=1)
    #cr_data_count = cosmic_ray_like_count(data_xyoff_map_1d,logE,region_type=3)
    run_sr_chi2 = cosmic_ray_like_chi2(fit_params,big_eigenvectors,data_xyoff_map_1d,1)
    run_cr_chi2 = cosmic_ray_like_chi2(fit_cr_params,big_eigenvectors,data_xyoff_map_1d,3)

    sr_data_count = cosmic_ray_like_count(data_xyoff_map_1d,region_type=1)
    fit_data_count = cosmic_ray_like_count(fit_xyoff_map_1d,region_type=1)
    cr_data_count = cosmic_ray_like_count(data_xyoff_map_1d,region_type=3)
    fit_cr_data_count = cosmic_ray_like_count(fit_xyoff_map_1d,region_type=3)

    cr_qual = run_cr_chi2/cr_data_count
    sr_qual = sr_data_count/fit_data_count



    idx_1d = 0
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d += 1

                    cr_correct = 1.

                    data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += data_xyoff_map_1d[idx_1d-1]
                    fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += fit_xyoff_map_1d[idx_1d-1]*cr_correct

    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    original_ratio = 1.
                    #if data_xyoff_map[logE].waxis[idx_x,idx_y,gcut]==0.: continue
                    #original_ratio = fit_xyoff_map[logE].waxis[idx_x,idx_y,0]/data_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
                    if fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut]==0.: continue
                    original_ratio = fit_xyoff_map[logE].waxis[idx_x,idx_y,0]/fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
                    ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = original_ratio

    for gcut in range(1,gcut_bins):
        for logE in range(0,logE_nbins):

            if xoff_bins[logE]==1: continue
            if yoff_bins[logE]==1: continue

            avg_ratio = 0.
            count = 0.
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    avg_ratio += ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
                    count += 1.
            if count==0.: continue
            avg_ratio = avg_ratio/count

            rms_ratio = 0.
            count = 0.
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    rms_ratio += pow(ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut]-avg_ratio,2)
                    count += 1.
            if count==0.: continue
            rms_ratio = pow(rms_ratio/count,0.5)

            new_avg_ratio = 0.
            count = 0.
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    if rms_ratio==0.: continue
                    deviation = (ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut] - avg_ratio)/rms_ratio
                    if abs(deviation)<2.:
                        new_avg_ratio += ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
                        count += 1.
            if count==0.: continue
            avg_ratio = new_avg_ratio/count

            new_rms_ratio = 0.
            count = 0.
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    if rms_ratio==0.: continue
                    deviation = (ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut] - avg_ratio)/rms_ratio
                    if abs(deviation)<2.:
                        new_rms_ratio += pow(ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut]-avg_ratio,2)
                        count += 1.
            if count==0.: continue
            rms_ratio = pow(new_rms_ratio/count,0.5)


            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    if rms_ratio==0.: continue
                    deviation = (ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut] - avg_ratio)/rms_ratio
                    if abs(deviation)>5.:
                        ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = avg_ratio

    run_count = 0
    for run_number in runlist:

        print (f'analyzing {run_count}/{len(runlist)} runs...')
    
        print (f'analyzing run {run_number}')
        rootfile_name = f'{smi_input}/{run_number}.anasum.root'
        print (rootfile_name)
        if not os.path.exists(rootfile_name):
            print (f'file does not exist.')
            continue
        run_count += 1
    
        InputFile = ROOT.TFile(rootfile_name)

        TreeName = f'run_{run_number}/stereo/pointingDataReduced'
        TelTree = InputFile.Get(TreeName)
        TelTree.GetEntry(int(float(TelTree.GetEntries())/2.))
        TelRAJ2000 = TelTree.TelRAJ2000*180./np.pi
        TelDecJ2000 = TelTree.TelDecJ2000*180./np.pi
        TelElevation = TelTree.TelElevation
        TelAzimuth = TelTree.TelAzimuth
        bright_star_coord = GetBrightStars(TelRAJ2000,TelDecJ2000)
        gamma_source_coord = GetGammaSources(TelRAJ2000,TelDecJ2000)

        TreeName = f'run_{run_number}/stereo/DL3EventTree'
        EvtTree = InputFile.Get(TreeName)
        total_entries = EvtTree.GetEntries()
        print (f'total_entries = {total_entries}')
        EvtTree.GetEntry(0)
        time_start = EvtTree.timeOfDay
        EvtTree.GetEntry(total_entries-1)
        time_end = EvtTree.timeOfDay
        exposure_hours += (time_end-time_start)/3600.
        avg_tel_elev += TelElevation*(time_end-time_start)/3600.
        avg_tel_azim += TelAzimuth*(time_end-time_start)/3600.
        for entry in range(0,total_entries):
            EvtTree.GetEntry(entry)
            Xoff = EvtTree.Xoff
            Yoff = EvtTree.Yoff
            Xderot = EvtTree.Xderot
            Yderot = EvtTree.Yderot
            MSCW = EvtTree.MSCW/MSCW_cut
            MSCL = EvtTree.MSCL/MSCL_cut
            GammaCut = EventGammaCut(MSCL,MSCW)
            Energy = EvtTree.Energy
            NImages = EvtTree.NImages
            EmissionHeight = EvtTree.EmissionHeight
            Xcore = EvtTree.XCore
            Ycore = EvtTree.YCore
            Roff = pow(Xoff*Xoff+Yoff*Yoff,0.5)
            Rcore = pow(Xcore*Xcore+Ycore*Ycore,0.5)
            logE = logE_axis.get_bin(np.log10(Energy))
            if logE<0: continue
            if logE>=len(data_sky_map): continue
            if NImages<min_NImages: continue
            if EmissionHeight>max_EmissionHeight_cut: continue
            if EmissionHeight<min_EmissionHeight_cut: continue
            if Roff>max_Roff: continue
            if Rcore>max_Rcore: continue
            if Rcore<min_Rcore: continue
            if Energy<min_Energy_cut: continue
            if Energy>max_Energy_cut: continue
            #if not control_region:
            #    if MSCL>1.0: continue
            #else:
            #    if MSCL<1.0: continue
            #    if MSCL>2.0: continue
            if GammaCut>float(gcut_end): continue

            Xsky = TelRAJ2000 + Xderot
            Ysky = TelDecJ2000 + Yderot

            cr_correction = ratio_xyoff_map[logE].get_bin_content(Xoff,Yoff,GammaCut)
            if GammaCut<1.:
                cr_correction = 1.

            data_sky_map[logE].fill(Xsky,Ysky,GammaCut)
            fit_sky_map[logE].fill(Xsky,Ysky,GammaCut,weight=cr_correction)
    
        print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')

        InputFile.Close()
  
    tracemalloc.stop()
    if exposure_hours>0.:
        avg_tel_elev = avg_tel_elev/exposure_hours
        avg_tel_azim = avg_tel_azim/exposure_hours

    return [exposure_hours,avg_tel_elev,avg_tel_azim,truth_params,fit_params,sr_qual,cr_qual], data_sky_map, fit_sky_map, data_xyoff_map, fit_xyoff_map


#def cosmic_ray_like_chi2(try_params,eigenvectors,xyoff_map,logE,region_type):
def cosmic_ray_like_chi2(try_params,eigenvectors,xyoff_map,region_type):

    try_params = np.array(try_params)
    try_xyoff_map = eigenvectors.T @ try_params

    sum_log_likelihood = 0.
    idx_1d = 0
    nbins = 0.
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            n_expect_total = 0.
            n_data_total = 0.
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d += 1
                    if region_type==0:
                        if gcut==0: continue
                    elif region_type==1:
                        if gcut!=0: continue
                    elif region_type==2:
                        if gcut==0: continue
                        if gcut==2: continue
                    elif region_type==3:
                        if gcut!=2: continue

                    weight = 1.
                    #weight = 1./float(gcut+1)

                    n_expect_total += try_xyoff_map[idx_1d-1]
                    n_data_total += xyoff_map[idx_1d-1]

                    #n_expect = max(0.0001,try_xyoff_map[idx_1d-1])
                    #n_data = xyoff_map[idx_1d-1]
                    #if n_data==0.:
                    #    sum_log_likelihood += n_expect
                    #else:
                    #    sum_log_likelihood += -1.*(n_data*np.log(n_expect) - n_expect - (n_data*np.log(n_data)-n_data))
                    #nbins += 1.

                    n_expect = try_xyoff_map[idx_1d-1]
                    n_data = xyoff_map[idx_1d-1]
                    sum_log_likelihood += pow(n_expect-n_data,2)*weight

    #return sum_log_likelihood/nbins
    return sum_log_likelihood

def cosmic_ray_like_count(xyoff_map,region_type=0):

    count = 0.
    idx_1d = 0
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d += 1
                    if region_type==0:
                        if gcut==0: continue
                    elif region_type==1:
                        if gcut!=0: continue
                    elif region_type==2:
                        if gcut==0: continue
                        if gcut==2: continue
                    elif region_type==3:
                        if gcut!=2: continue

                    weight = 1.
                    #weight = 1./float(gcut+1)

                    count += xyoff_map[idx_1d-1]*weight

    return count

def ReadSNRTargetListFromCSVFile():
    source_name = []
    source_ra = []
    source_dec = []
    source_size = []
    with open('SNRcat20221001-SNR.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if len(row)==0: continue
            if '#' in row[0]: continue
            target_name = row[0]
            target_min_dist = row[13]
            if target_min_dist=='':
                target_min_dist = '1000'
            #if float(target_min_dist)>6.: continue
            target_size = row[15]
            if target_size=='':
                target_size = 0.
            target_ra = row[19]
            target_dec = row[20]
            source_name += [target_name]
            source_ra += [float(HMS2deg(target_ra,target_dec)[0])]
            source_dec += [float(HMS2deg(target_ra,target_dec)[1])]
            source_size += [0.5*float(target_size)/60.]
            #print('target_min_dist = %s'%(target_min_dist))
            #print('source_name = %s'%(source_name[len(source_name)-1]))
            #print('source_ra = %0.2f'%(source_ra[len(source_ra)-1]))
            #print('source_dec = %0.2f'%(source_dec[len(source_dec)-1]))
            #print(row)
    return source_name, source_ra, source_dec, source_size

def HMS2deg(ra='', dec=''):
    RA, DEC, rs, ds = '', '', 1, 1
    if dec:
        D, M, S = [float(i) for i in dec.split(':')]
        if str(D)[0] == '-':
            ds, D = -1, abs(D)
        deg = D + (M/60) + (S/3600)
        DEC = '{0}'.format(deg*ds)
    if ra:
        H, M, S = [float(i) for i in ra.split(':')]
        if str(H)[0] == '-':
            rs, H = -1, abs(H)
        deg = (H*15) + (M/4) + (S/240)
        RA = '{0}'.format(deg*rs)           
    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC

def ReadATNFTargetListFromFile(file_path):
    source_name = []
    source_ra = []
    source_dec = []
    source_dist = []
    source_age = []
    source_edot = []
    inputFile = open(file_path)
    for line in inputFile:
        if line[0]=="#": continue
        target_name = line.split(',')[0].strip(" ")
        if target_name=="\n": continue
        target_ra = line.split(',')[1].strip(" ")
        target_dec = line.split(',')[2].strip(" ")
        target_dist = line.split(',')[3].strip(" ")
        target_age = line.split(',')[4].strip(" ")
        target_edot = line.split(',')[5].strip(" ")
        if target_dist=='*': continue
        if target_age=='*': continue
        if target_edot=='*': continue
        target_brightness = float(target_edot)/pow(float(target_dist),2)

        if float(target_edot)<1e35: continue
        #if float(target_dist)>6.: continue

        #ra_deg = float(HMS2deg(target_ra,target_dec)[0])
        #dec_deg = float(HMS2deg(target_ra,target_dec)[1])
        #gal_l, gal_b = ConvertRaDecToGalactic(ra_deg,dec_deg)
        #if abs(gal_b)<5.: continue

        source_name += [target_name]
        source_ra += [float(HMS2deg(target_ra,target_dec)[0])]
        source_dec += [float(HMS2deg(target_ra,target_dec)[1])]
        source_dist += [float(target_dist)]
        source_age += [float(target_age)]
        source_edot += [float(target_edot)]
    return source_name, source_ra, source_dec, source_dist, source_age

def ReadBrightStarListFromFile():
    star_name = []
    star_ra = []
    star_dec = []
    inputFile = open('Hipparcos_MAG8_1997.dat')
    for line in inputFile:
        if line[0]=="#": continue
        if line[0]=="*": continue
        if len(line.split())<5: continue
        target_ra = float(line.split()[0])
        target_dec = float(line.split()[1])
        target_brightness = float(line.split()[3])+float(line.split()[4])
        if target_brightness>7.: continue
        #if target_brightness>6.: continue
        #if target_brightness>5.: continue
        star_ra += [target_ra]
        star_dec += [target_dec]
        star_name += ['bmag = %0.1f'%(target_brightness)]
    #print ('Found %s bright stars.'%(len(star_name)))
    return star_name, star_ra, star_dec

def ReadHAWCTargetListFromFile(file_path):
    source_name = []
    source_ra = []
    source_dec = []
    inputFile = open(file_path)
    for line in inputFile:
        if line[0]=="#": continue
        if '- name:' in line:
            target_name = line.lstrip('   - name: ')
            target_name = target_name.strip('\n')
        if 'RA:' in line:
            target_ra = line.lstrip('     RA: ')
        if 'Dec:' in line:
            target_dec = line.lstrip('     Dec: ')
        if 'flux measurements:' in line:
            source_name += [target_name]
            source_ra += [float(target_ra)]
            source_dec += [float(target_dec)]
            target_name = ''
            target_ra = ''
            target_dec = ''
    return source_name, source_ra, source_dec

def ReadLhaasoListFromFile():
    source_name = []
    source_ra = []
    source_dec = []
    file_path = 'LHAASO_1st_Catalog.txt'
    inputFile = open(file_path)
    for line in inputFile:
        source_name += ['%s %s'%(line.split()[0],line.split()[1])]
        source_ra += [float(line.split()[3])]
        source_dec += [float(line.split()[4])]
    return source_name, source_ra, source_dec

def ReadFermiCatelog():
    source_name = []
    source_ra = []
    source_dec = []
    inputFile = open('gll_psc_v26.xml')
    target_name = ''
    target_type = ''
    target_info = ''
    target_flux = ''
    target_ra = ''
    target_dec = ''
    for line in inputFile:
        if line.split(' ')[0]=='<source':
            for block in range(0,len(line.split(' '))):
                if 'Unc_' in line.split(' ')[block]: continue
                if 'name=' in line.split(' ')[block]:
                    target_name = line.split('name="')[1].split('"')[0]
                if 'type=' in line.split(' ')[block]:
                    target_type = line.split('type="')[1].split('"')[0]
                if 'Flux1000=' in line.split(' ')[block]:
                    target_flux = line.split('Flux1000="')[1].split('"')[0]
                if 'Energy_Flux100=' in line.split(' ')[block]:
                    target_info = line.split(' ')[block]
                    target_info = target_info.strip('>\n')
                    target_info = target_info.strip('"')
                    target_info = target_info.lstrip('Energy_Flux100="')
        if '<parameter' in line and 'name="RA"' in line:
            for block in range(0,len(line.split(' '))):
                if 'value=' in line.split(' ')[block]:
                    target_ra = line.split(' ')[block].split('"')[1]
        if '<parameter' in line and 'name="DEC"' in line:
            for block in range(0,len(line.split(' '))):
                if 'value=' in line.split(' ')[block]:
                    target_dec = line.split(' ')[block].split('"')[1]
        if 'source>' in line:
            if target_ra=='': 
                target_name = ''
                target_type = ''
                target_info = ''
                target_ra = ''
                target_dec = ''
                continue
            #if target_type=='PointSource': 
            #if float(target_info)<1e-5: 
            if float(target_flux)<1e-9: 
                target_name = ''
                target_type = ''
                target_info = ''
                target_ra = ''
                target_dec = ''
                continue
            #print ('target_name = %s'%(target_name))
            #print ('target_type = %s'%(target_type))
            #print ('target_ra = %s'%(target_ra))
            #print ('target_dec = %s'%(target_dec))
            source_name += [target_name]
            #source_name += ['%0.2e'%(float(target_info))]
            source_ra += [float(target_ra)]
            source_dec += [float(target_dec)]
            target_name = ''
            target_type = ''
            target_ra = ''
            target_dec = ''
    return source_name, source_ra, source_dec


def GetGammaSourceInfo():

    other_stars = []
    other_stars_type = []
    other_star_coord = []

    #return other_stars, other_stars_type, other_star_coord

    near_source_cut = 0.1

    drawBrightStar = False
    drawPulsar = True
    drawSNR = True
    drawLHAASO = False
    drawFermi = False
    drawHAWC = False
    drawTeV = False

    if drawBrightStar:
        star_name, star_ra, star_dec = ReadBrightStarListFromFile()
        for src in range(0,len(star_name)):
            src_ra = star_ra[src]
            src_dec = star_dec[src]
            other_stars += [star_name[src]]
            other_stars_type += ['Star']
            other_star_coord += [[src_ra,src_dec,0.]]

    if drawSNR:
        target_snr_name, target_snr_ra, target_snr_dec, target_snr_size = ReadSNRTargetListFromCSVFile()
        for src in range(0,len(target_snr_name)):
            gamma_source_name = target_snr_name[src]
            gamma_source_ra = target_snr_ra[src]
            gamma_source_dec = target_snr_dec[src]
            gamma_source_size = target_snr_size[src]
            near_a_source = False
            for entry in range(0,len(other_stars)):
                distance = pow(gamma_source_ra-other_star_coord[entry][0],2)+pow(gamma_source_dec-other_star_coord[entry][1],2)
                if distance<near_source_cut*near_source_cut:
                    near_a_source = True
            if not near_a_source:
                other_stars += [gamma_source_name]
                other_stars_type += ['SNR']
                other_star_coord += [[gamma_source_ra,gamma_source_dec,gamma_source_size]]

    if drawPulsar:
        target_psr_name, target_psr_ra, target_psr_dec, target_psr_dist, target_psr_age = ReadATNFTargetListFromFile('ATNF_pulsar_full_list.txt')
        for src in range(0,len(target_psr_name)):
            gamma_source_name = target_psr_name[src]
            gamma_source_ra = target_psr_ra[src]
            gamma_source_dec = target_psr_dec[src]
            near_a_source = False
            for entry in range(0,len(other_stars)):
                distance = pow(gamma_source_ra-other_star_coord[entry][0],2)+pow(gamma_source_dec-other_star_coord[entry][1],2)
                if distance<near_source_cut*near_source_cut:
                    near_a_source = True
            if not near_a_source:
                other_stars += [gamma_source_name]
                if target_psr_age[src]/1000.>1e6:
                        other_stars_type += ['MSP']
                else:
                        other_stars_type += ['PSR']
                other_star_coord += [[gamma_source_ra,gamma_source_dec,0.]]

    if drawHAWC:
        target_hwc_name, target_hwc_ra, target_hwc_dec = ReadHAWCTargetListFromFile('Cat_3HWC.txt')
        for src in range(0,len(target_hwc_name)):
            gamma_source_name = target_hwc_name[src]
            gamma_source_ra = target_hwc_ra[src]
            gamma_source_dec = target_hwc_dec[src]
            near_a_source = False
            for entry in range(0,len(other_stars)):
                distance = pow(gamma_source_ra-other_star_coord[entry][0],2)+pow(gamma_source_dec-other_star_coord[entry][1],2)
                if distance<near_source_cut*near_source_cut:
                    near_a_source = True
            if not near_a_source:
                other_stars += [gamma_source_name]
                other_stars_type += ['HAWC']
                other_star_coord += [[gamma_source_ra,gamma_source_dec,0.]]

    if drawLHAASO:
        target_lhs_name, target_lhs_ra, target_lhs_dec = ReadLhaasoListFromFile()
        for src in range(0,len(target_lhs_name)):
            gamma_source_name = target_lhs_name[src]
            gamma_source_ra = target_lhs_ra[src]
            gamma_source_dec = target_lhs_dec[src]
            near_a_source = False
            for entry in range(0,len(other_stars)):
                distance = pow(gamma_source_ra-other_star_coord[entry][0],2)+pow(gamma_source_dec-other_star_coord[entry][1],2)
                if distance<near_source_cut*near_source_cut:
                    near_a_source = True
            if not near_a_source:
                other_stars += [gamma_source_name]
                other_stars_type += ['LHAASO']
                other_star_coord += [[gamma_source_ra,gamma_source_dec,0.]]

    if drawFermi:
        fermi_name, fermi_ra, fermi_dec = ReadFermiCatelog()
        for src in range(0,len(fermi_name)):
            gamma_source_name = fermi_name[src]
            gamma_source_ra = fermi_ra[src]
            gamma_source_dec = fermi_dec[src]
            near_a_source = False
            for entry in range(0,len(other_stars)):
                distance = pow(gamma_source_ra-other_star_coord[entry][0],2)+pow(gamma_source_dec-other_star_coord[entry][1],2)
                if distance<near_source_cut*near_source_cut:
                    near_a_source = True
            if not near_a_source:
                other_stars += [gamma_source_name]
                other_stars_type += ['Fermi']
                other_star_coord += [[gamma_source_ra,gamma_source_dec,0.]]

    if drawTeV:
        inputFile = open('TeVCat_RaDec_w_Names.txt')
        for line in inputFile:
            gamma_source_name = line.split(',')[0]
            gamma_source_ra = float(line.split(',')[1])
            gamma_source_dec = float(line.split(',')[2])
            near_a_source = False
            for entry in range(0,len(other_stars)):
                distance = pow(gamma_source_ra-other_star_coord[entry][0],2)+pow(gamma_source_dec-other_star_coord[entry][1],2)
                if distance<near_source_cut*near_source_cut:
                    near_a_source = True
            if not near_a_source and not '%' in gamma_source_name:
                other_stars += [gamma_source_name]
                other_stars_type += ['TeV']
                other_star_coord += [[gamma_source_ra,gamma_source_dec,0.]]

    return other_stars, other_stars_type, other_star_coord


def PlotSkyMap(fig,hist_map,plotname,roi_x=[],roi_y=[],roi_r=[],max_z=0.,colormap='coolwarm',layer=0):

    xmin = hist_map.xaxis.min()
    xmax = hist_map.xaxis.max()
    ymin = hist_map.yaxis.min()
    ymax = hist_map.yaxis.max()

    other_stars, other_star_type, other_star_coord = GetGammaSourceInfo() 

    other_star_labels = []
    other_star_types = []
    other_star_markers = []
    star_range = 0.8*(xmax-xmin)/2.
    source_ra = (xmax+xmin)/2.
    source_dec = (ymax+ymin)/2.
    #print (f'star_range = {star_range}')
    #print (f'source_ra = {source_ra}')
    #print (f'source_dec = {source_dec}')
    n_stars = 0
    for star in range(0,len(other_stars)):
        if abs(source_ra-other_star_coord[star][0])>star_range: continue
        if abs(source_dec-other_star_coord[star][1])>star_range: continue
        other_star_markers += [[other_star_coord[star][0],other_star_coord[star][1],other_star_coord[star][2]]]
        other_star_labels += ['%s'%(other_stars[star])]
        other_star_types += [other_star_type[star]]
        n_stars += 1
    for star in range(0,len(other_star_markers)):
        print (f'Star {other_star_labels[star]} RA = {other_star_markers[star][0]:0.1f}, Dec = {other_star_markers[star][1]:0.1f}')

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'RA [deg]'
    label_y = 'Dec [deg]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(hist_map.waxis[:,:,layer].T,origin='lower',extent=(xmax,xmin,ymin,ymax),aspect='auto',cmap='coolwarm')
    if max_z!=0.:
        im = axbig.imshow(hist_map.waxis[:,:,layer].T,origin='lower',extent=(xmax,xmin,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
    cbar = fig.colorbar(im)

    font = {'family': 'serif', 'color':  'k', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

    favorite_color = 'k'
    if colormap=='gray':
        favorite_color = 'r'
    if colormap=='magma':
        favorite_color = 'g'
    for star in range(0,len(other_star_markers)):
        marker_size = 60
        if other_star_types[star]=='PSR':
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=1.5*marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='SNR':
            mycircle = plt.Circle( (other_star_markers[star][0], other_star_markers[star][1]), other_star_markers[star][2], fill = False, color=favorite_color)
            axbig.add_patch(mycircle)
        if other_star_types[star]=='LHAASO':
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='HAWC':
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='Fermi':
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='MSP':
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c='tomato', marker='+', label=other_star_labels[star])
        if other_star_types[star]=='TeV':
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c='lime', marker='+', label=other_star_labels[star])
        if other_star_types[star]=='Star':
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c='k', marker='+', label=other_star_labels[star])
        txt = axbig.text(other_star_markers[star][0]-0.07, other_star_markers[star][1]+0.07, other_star_labels[star], fontdict=font)

    fig.savefig(f'output_plots/{plotname}.png',bbox_inches='tight')
    axbig.remove()

def GetFluxCalibration(energy):

    if doFluxCalibration:
        return 1.

    str_flux_calibration = ['2.23e+01', '3.12e+01', '3.17e+01', '1.38e+01', '1.24e+01', '1.28e+01', '1.35e+01', '1.32e+01']

    flux_calibration = []
    for string in str_flux_calibration:
        flux_calibration.append(float(string))

    return 1./flux_calibration[energy]

def make_significance_map(data_sky_map,bkgd_sky_map,significance_sky_map,excess_sky_map):
  
    skymap_bins = len(data_sky_map.xaxis)-1

    gcut_weight = []
    for gcut in range(0,gcut_bins):
        gcut_weight += [pow(np.sum(bkgd_sky_map.waxis[:,:,gcut]),1.0)]

    for idx_x in range(0,skymap_bins):
        for idx_y in range(0,skymap_bins):
            data = data_sky_map.waxis[idx_x,idx_y,0]
            bkgd = 0.
            gcut_norm = 0.
            for gcut in range(1,gcut_bins):
                gcut_norm += gcut_weight[gcut]
            for gcut in range(1,gcut_bins):
               bkgd += gcut_weight[gcut]/gcut_norm*bkgd_sky_map.waxis[idx_x,idx_y,gcut]
            #bkgd = bkgd_sky_map.waxis[idx_x,idx_y,1]
            data_err = pow(data,0.5)
            if data_err==0.: continue
            significance_sky_map.waxis[idx_x,idx_y,0] = (data-bkgd)/data_err
            excess_sky_map.waxis[idx_x,idx_y,0] = (data-bkgd)

def make_flux_map(data_sky_map,bkgd_sky_map,flux_sky_map,flux_err_sky_map,avg_energy,delta_energy):
  
    skymap_bins = len(data_sky_map.xaxis)-1

    gcut_weight = []
    for gcut in range(0,gcut_bins):
        gcut_weight += [pow(np.sum(bkgd_sky_map.waxis[:,:,gcut]),1.0)]

    norm_content_max = 0.
    for idx_x in range(0,skymap_bins):
        for idx_y in range(0,skymap_bins):
            norm = bkgd_sky_map.waxis[idx_x,idx_y,0]
            if norm_content_max<norm:
                norm_content_max = norm

    for idx_x in range(0,skymap_bins):
        for idx_y in range(0,skymap_bins):
            data = data_sky_map.waxis[idx_x,idx_y,0]
            norm = bkgd_sky_map.waxis[idx_x,idx_y,0]
            bkgd = 0.
            gcut_norm = 0.
            for gcut in range(1,gcut_bins):
                gcut_norm += gcut_weight[gcut]
            for gcut in range(1,gcut_bins):
                bkgd += gcut_weight[gcut]/gcut_norm*bkgd_sky_map.waxis[idx_x,idx_y,gcut]
            #bkgd = bkgd_sky_map.waxis[idx_x,idx_y,1]
            if norm>0.:
                excess = data-bkgd
                error = max(pow(data,0.5),1.)
                logE = logE_axis.get_bin(np.log10(avg_energy))
                correction = GetFluxCalibration(logE)/norm*pow(avg_energy,2)/(100.*100.*3600.)/delta_energy
                norm_ratio = norm/norm_content_max
                norm_weight = 1./(1.+np.exp(-(norm_ratio-0.3)/0.05))
                flux = excess*correction*norm_weight
                flux_err = error*correction*norm_weight
                flux_sky_map.waxis[idx_x,idx_y,0] = flux
                flux_err_sky_map.waxis[idx_x,idx_y,0] = flux_err
            else:
                flux_sky_map.waxis[idx_x,idx_y,0] = 0.
                flux_err_sky_map.waxis[idx_x,idx_y,0] = 0.

def GetRadialProfile(hist_flux_skymap,hist_error_skymap,roi_x,roi_y,roi_r):

    radial_axis = MyArray1D(x_nbins=10,start_x=0.,end_x=roi_r)

    radius_array = []
    brightness_array = []
    brightness_err_array = []
    pixel_array = []
    for br in range(0,len(radial_axis.xaxis)-1):
        radius = 0.5*(radial_axis.xaxis[br]+radial_axis.xaxis[br+1])
        radius_array += [radius]
        brightness_array += [0.]
        brightness_err_array += [0.]
        pixel_array += [0.]

    for br in range(0,len(radial_axis.xaxis)-1):
        radius = 0.5*(radial_axis.xaxis[br]+radial_axis.xaxis[br+1])
        for bx in range(0,len(hist_flux_skymap.xaxis)-1):
            for by in range(0,len(hist_flux_skymap.yaxis)-1):
                bin_ra = 0.5*(hist_flux_skymap.xaxis[bx]+hist_flux_skymap.xaxis[bx+1])
                bin_dec = 0.5*(hist_flux_skymap.yaxis[by]+hist_flux_skymap.yaxis[by+1])
                keep_event = False
                distance = pow(pow(bin_ra-roi_x,2) + pow(bin_dec-roi_y,2),0.5)
                if distance<radial_axis.xaxis[br+1] and distance>=radial_axis.xaxis[br]: 
                    keep_event = True
                if keep_event:
                    pixel_array[br] += 1.
                    if not hist_flux_skymap.waxis[bx,by,0]==0.:
                        brightness_array[br] += hist_flux_skymap.waxis[bx,by,0]
                        brightness_err_array[br] += pow(hist_error_skymap.waxis[bx,by,0],2)
                    else:
                        brightness_array[br] += 0.
                        brightness_err_array[br] = max(pow(hist_error_skymap.waxis[bx,by,0],2),brightness_err_array[br])
        if pixel_array[br]==0.: continue
        brightness_array[br] = brightness_array[br]/pixel_array[br]
        brightness_err_array[br] = pow(brightness_err_array[br],0.5)/pixel_array[br]

    return radius_array, brightness_array, brightness_err_array

def GetRegionIntegral(hist_flux_skymap,hist_error_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r):

    flux_sum = 0.
    flux_stat_err = 0.
    for bx in range(0,len(hist_flux_skymap.xaxis)-1):
        for by in range(0,len(hist_flux_skymap.yaxis)-1):
            bin_ra = 0.5*(hist_flux_skymap.xaxis[bx]+hist_flux_skymap.xaxis[bx+1])
            bin_dec = 0.5*(hist_flux_skymap.yaxis[by]+hist_flux_skymap.yaxis[by+1])
            keep_event = False
            for roi in range(0,len(roi_x)):
                distance = pow(pow(bin_ra-roi_x[roi],2) + pow(bin_dec-roi_y[roi],2),0.5)
                if distance<roi_r[roi]: 
                    keep_event = True
            for roi in range(0,len(excl_roi_x)):
                excl_distance = pow(pow(bin_ra-excl_roi_x[roi],2) + pow(bin_dec-excl_roi_y[roi],2),0.5)
                if excl_distance<excl_roi_r[roi]: 
                    keep_event = False
            if keep_event:
                if not hist_flux_skymap.waxis[bx,by,0]==0.:
                    flux_sum += hist_flux_skymap.waxis[bx,by,0]
                    flux_stat_err += pow(hist_error_skymap.waxis[bx,by,0],2)
                else:
                    flux_sum += 0.
                    flux_stat_err = max(pow(hist_error_skymap.waxis[bx,by,0],2),flux_stat_err)
    flux_stat_err = pow(flux_stat_err,0.5)
    return flux_sum, flux_stat_err

def GetRegionSpectrum(hist_flux_skymap,hist_error_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r):

    x_axis = []
    x_error = []
    y_axis = []
    y_error = []

    binE_start = 0
    binE_end = logE_nbins

    for binE in range(binE_start,binE_end):
        flux_sum = 0.
        flux_stat_err = 0.
        flux_syst_err = 0.
        flux_sum, flux_stat_err = GetRegionIntegral(hist_flux_skymap[binE],hist_error_skymap[binE],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)
        x_axis += [0.5*(pow(10.,logE_axis.xaxis[binE+1])+pow(10.,logE_axis.xaxis[binE]))]
        x_error += [0.5*(pow(10.,logE_axis.xaxis[binE+1])-pow(10.,logE_axis.xaxis[binE]))]
        y_axis += [flux_sum]
        y_error += [flux_stat_err]

    return x_axis, x_error, y_axis, y_error

def flux_crab_func(x):
    # TeV^{-1}cm^{-2}s^{-1}
    # Crab https://arxiv.org/pdf/1508.06442.pdf
    return 37.5*pow(10,-12)*pow(x*1./1000.,-2.467-0.16*np.log(x/1000.))

def PrintFluxCalibration(fig,hist_flux_skymap,hist_error_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r):

    energy_axis, energy_error, flux, flux_stat_err = GetRegionSpectrum(hist_flux_skymap,hist_error_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)
    print (f'energy_axis = {energy_axis}')

    vectorize_f_crab = np.vectorize(flux_crab_func)
    ydata_crab_ref = pow(np.array(energy_axis),2)*vectorize_f_crab(energy_axis)

    #log_energy = np.linspace(np.log10(2e2),np.log10(1.2e4),50)
    #xdata = pow(10.,log_energy)
    #ydata_crab = pow(xdata/1e3,2)*vectorize_f_crab(xdata)

    calibration_new = []
    for binx in range(0,len(energy_axis)):
        if flux[binx]>0.:
            calibration_new += [flux[binx]/ydata_crab_ref[binx]]
        else:
            calibration_new += [0.]
    print ('=======================================================================')
    formatted_numbers = ['%0.2e' % num for num in calibration_new]
    print ('new flux_calibration = %s'%(formatted_numbers))
    print ('=======================================================================')

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Energy [TeV]'
    label_y = 'Flux in C.U.'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.set_xscale('log')
    axbig.set_yscale('log')
    axbig.plot(energy_axis, calibration_new, color='b', ls='dashed')
    fig.savefig(f'output_plots/flux_crab_unit.png',bbox_inches='tight')
    axbig.remove()

def DefineRegionOfInterest(src_name,src_ra,src_dec):

    excl_region_x = [src_ra]
    excl_region_y = [src_dec]
    excl_region_r = [0.0]
    region_x = [src_ra]
    region_y = [src_dec]
    region_r = [calibration_radius]

    if 'Crab' in src_name:
        region_x = [src_ra]
        region_y = [src_dec]
        region_r = [calibration_radius]

    return region_x, region_y, region_r, excl_region_x, excl_region_y, excl_region_r

