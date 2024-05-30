
import os, sys
import ROOT
import numpy as np
import pickle
import csv
from scipy.optimize import least_squares, minimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tracemalloc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import wcs
from astropy.io import fits

sky_tag = os.environ.get("SKY_TAG")


use_poisson_likelihood = True
fix_init_scale = 0.
matrix_rank = 30

if sky_tag=='linear':
    use_poisson_likelihood = False
if sky_tag=='poisson':
    use_poisson_likelihood = True

if sky_tag=='init':
    fix_init_scale = 1e5

if 'rank' in sky_tag:
    matrix_rank = int(sky_tag.strip('rank'))

run_elev_cut = 25.

min_NImages = 2
max_Roff = 1.7
max_EmissionHeight_cut = 20.
min_EmissionHeight_cut = 6.
max_Rcore = 400.
min_Rcore = 0.
min_Energy_cut = 0.2
max_Energy_cut = 10.0
MVA_cut = 0.5

xoff_start = -2.
xoff_end = 2.
yoff_start = -2.
yoff_end = 2.
gcut_bins = 4
gcut_start = 0
gcut_end = gcut_bins

logE_bins = [-0.625,-0.600,-0.50,-0.375,-0.25,0.00,0.25,0.50,0.75,1.0] # logE TeV
#logE_bins = [-0.75,-0.625,-0.50,-0.375,-0.25,0.00,0.25,0.50,0.75,1.0] # logE TeV
logE_nbins = len(logE_bins)-1

#MSCW_cut = 0.7
#MSCL_cut = 0.8
MSCW_cut = [0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]
MSCL_cut = [0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10]

skymap_size = 3.
skymap_bins = 30
fine_skymap_bins = 120

#doFluxCalibration = True
doFluxCalibration = False
calibration_radius = 0.15 # need to be larger than the PSF and smaller than the integration radius

#logE_min = 3
#logE_mid = 7
#logE_max = logE_nbins
#xoff_bins = [11,11,11,11,11,11,11,11,11]
xoff_bins = [11,11,9,7,5,3,3,1,1]
#xoff_bins = [11,9,5,5,3,3,1,1]
#xoff_bins = [13,11,9,5,5,3,3,1,1]
yoff_bins = xoff_bins

chi2_cut = 0.5
#chi2_cut = 1e10

smi_aux = os.environ.get("SMI_AUX")
smi_dir = os.environ.get("SMI_DIR")

def GetRunElevAzim(smi_input,run_number):

    rootfile_name = f'{smi_input}/{run_number}.anasum.root'
    print (rootfile_name)
    if not os.path.exists(rootfile_name):
        print (f'file does not exist.')
        return 0, 0
    
    InputFile = ROOT.TFile(rootfile_name)
    TreeName = f'run_{run_number}/stereo/pointingDataReduced'
    TelTree = InputFile.Get(TreeName)
    TelTree.GetEntry(int(float(TelTree.GetEntries())/2.))
    TelElevation = TelTree.TelElevation
    TelAzimuth = TelTree.TelAzimuth

    return TelElevation, TelAzimuth

def sortFirst(val):
    return val[0]

def ReadRunListFromFile(smi_input,input_on_file,input_off_file,input_mimic_file):

    on_runlist = []
    off_runlist = []
    mimic_runlist = []
    all_runlist = []

    on_inputFile = open(input_on_file)
    for on_line in on_inputFile:

        onrun_elev, onrun_azim = GetRunElevAzim(smi_input,int(on_line))

        off_inputFile = open(input_off_file)
        paired_off_runs = []
        for off_line in off_inputFile:
            line_split = off_line.split()
            on_run = int(line_split[0])
            off_run = int(line_split[1])
            if on_run==int(on_line):
                paired_off_runs += [off_run]

        mimic_inputFile = open(input_mimic_file)
        paired_mimic_runs = []
        for mimic_line in mimic_inputFile:
            line_split = mimic_line.split()
            on_run = int(line_split[0])
            mimic_run = int(line_split[1])
            if on_run==int(on_line):
                paired_mimic_runs += [mimic_run]

        all_runlist += [(onrun_elev,int(on_line),paired_off_runs,paired_mimic_runs)]

    all_runlist.sort(key=sortFirst,reverse=True)
    #all_runlist.sort(key=sortFirst)

    for run in range(0,len(all_runlist)):
        on_runlist += [all_runlist[run][1]]
        off_runlist += [all_runlist[run][2]]
        mimic_runlist += [all_runlist[run][3]]

    return on_runlist, off_runlist, mimic_runlist

def ReadOffRunListFromFile(input_file):

    runlist = []

    inputFile = open(input_file)
    for line in inputFile:
        line_split = line.split()
        runlist += [int(line_split[1])]

    return runlist

def smooth_image(image_data,xaxis,yaxis,kernel_radius=0.07):

    #return

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
            for idx_x2 in range(idx_x1-3*kernel_pix_size,idx_x1+3*kernel_pix_size):
                for idx_y2 in range(idx_y1-3*kernel_pix_size,idx_y1+3*kernel_pix_size):
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

    def just_like(self, template):
        self.xaxis = np.zeros_like(template.xaxis)
        self.yaxis = np.zeros_like(template.yaxis)
        self.zaxis = np.zeros_like(template.zaxis)
        self.waxis = np.zeros_like(template.waxis)
        self.overflow = template.overflow
        self.delta_x = template.delta_x
        self.delta_y = template.delta_y
        self.delta_z = template.delta_z
        for idx in range(0,len(self.xaxis)):
            self.xaxis[idx] = template.xaxis[idx]
        for idx in range(0,len(self.yaxis)):
            self.yaxis[idx] = template.yaxis[idx]
        for idx in range(0,len(self.zaxis)):
            self.zaxis[idx] = template.zaxis[idx]

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
    #print (f'Found {len(bright_stars_coord)} Gamma-ray sources.')
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

    #print (f'Found {len(bright_stars_coord)} bright stars.')
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

def CoincideWithRegionOfInterest(ra, dec, roi_ra, roi_dec, roi_r):
    isCoincident = False
    for roi in range(0,len(roi_r)):
        distance = pow(pow(roi_ra[roi]-ra,2)+pow(roi_dec[roi]-dec,2),0.5)
        if distance>roi_r[roi]: continue
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
    elif abs(MSCL)<2. and abs(MSCW)<2.:
        GammaCut = 3.5

    #if abs(MSCL)<1. and abs(MSCW)<1.:
    #    GammaCut = 0.5
    #elif abs(MSCL)<1. and abs(MSCW)<2.:
    #    GammaCut = 1.5
    #elif abs(MSCL)<1. and abs(MSCW)<3.:
    #    GammaCut = 2.5
    #elif abs(MSCL)<1. and abs(MSCW)<4.:
    #    GammaCut = 3.5

    return GammaCut

def build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,runlist,max_runs=1e10,is_bkgd=True,is_on=False,specific_run=0):

    #big_matrix = []
    #for logE in range(0,logE_nbins):
    #    big_matrix += [None]

    big_matrix = []
    big_mask_matrix = []

    region_name = source_name
    if not is_on:
        region_name = 'Validation'
    roi_name,roi_ra,roi_dec,roi_r = DefineRegionOfInterest(region_name,src_ra,src_dec)

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
        xyoff_mask_map = []
        for logE in range(0,logE_nbins):
            xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
            xyoff_mask_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    
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
        
        if not is_bkgd:
            if TelElevation<run_elev_cut: continue

        TreeName = f'run_{run_number}/stereo/DL3EventTree'
        EvtTree = InputFile.Get(TreeName)
        total_entries = EvtTree.GetEntries()
        #print (f'total_entries = {total_entries}')
        for entry in range(0,total_entries):
            EvtTree.GetEntry(entry)
            Xoff = EvtTree.Xoff
            Yoff = EvtTree.Yoff
            Xderot = EvtTree.Xderot
            Yderot = EvtTree.Yderot
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
            MSCW = EvtTree.MSCW/MSCW_cut[logE]
            MSCL = EvtTree.MSCL/MSCL_cut[logE]
            GammaCut = EventGammaCut(MSCL,MSCW)
            if GammaCut>float(gcut_end): continue

            Xsky = TelRAJ2000 + Xderot
            Ysky = TelDecJ2000 + Yderot

            #mirror_Xsky = TelRAJ2000 - Xderot
            #mirror_Ysky = TelDecJ2000 - Yderot
            #found_bright_star = CoincideWithBrightStars(Xsky, Ysky, bright_star_coord)
            #found_gamma_source = CoincideWithBrightStars(Xsky, Ysky, gamma_source_coord)
            #found_mirror_star = CoincideWithBrightStars(mirror_Xsky, mirror_Ysky, bright_star_coord)
            #found_mirror_gamma_source = CoincideWithBrightStars(mirror_Xsky, mirror_Ysky, gamma_source_coord)
            #if is_bkgd:
            #    if found_bright_star: continue
            #    if found_gamma_source: continue
            #    if found_mirror_star or found_mirror_gamma_source:
            #        xyoff_map[logE].fill(-Xderot,-Yderot,GammaCut)

            if is_on:
                found_roi = CoincideWithRegionOfInterest(Xsky, Ysky, roi_ra, roi_dec, roi_r)
                if found_roi:
                    xyoff_mask_map[logE].fill(Xoff,Yoff,0.5)
            else:
                xyoff_mask_map[logE].fill(Xoff,Yoff,0.5)

            xyoff_map[logE].fill(Xoff,Yoff,GammaCut)
            #xyoff_map[logE].fill(Xderot,Yderot,GammaCut)
    
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

        xyoff_mask_map_1d = []
        for gcut in range(0,gcut_bins):
            for logE in range(0,logE_nbins):
                for idx_x in range(0,xoff_bins[logE]):
                    for idx_y in range(0,yoff_bins[logE]):
                        xyoff_mask_map_1d += [xyoff_mask_map[logE].waxis[idx_x,idx_y,gcut]]
        big_mask_matrix += [xyoff_mask_map_1d]


        InputFile.Close()
        if run_count==max_runs: break

    return big_matrix, big_mask_matrix

def build_skymap(source_name,src_ra,src_dec,smi_input,eigenvector_path,big_matrix_path,runlist,mimic_runlist,onoff,max_runs=1e10):

    global skymap_bins
    if onoff=='ON' or 'MIMIC' in onoff:
        skymap_bins = fine_skymap_bins

    # start memory profiling
    tracemalloc.start()

    xsky_start = src_ra+skymap_size
    xsky_end = src_ra-skymap_size
    ysky_start = src_dec-skymap_size
    ysky_end = src_dec+skymap_size

    print ('loading svd pickle data... ')
    input_filename = eigenvector_path
    eigen_stuff = pickle.load(open(input_filename, "rb"))
    big_eigenvalues = eigen_stuff[0]
    big_eigenvectors = eigen_stuff[1]
    avg_xyoff_map_1d = eigen_stuff[2]

    print ('loading matrix pickle data... ')
    input_filename = big_matrix_path
    big_matrix = pickle.load(open(input_filename, "rb"))


    exposure_hours = 0.
    avg_tel_elev = 0.
    avg_tel_azim = 0.
    incl_sky_map = []
    data_sky_map = []
    fit_sky_map = []
    for logE in range(0,logE_nbins):
        incl_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=1)]
        data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
        fit_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]

    data_xyoff_map = []
    fit_xyoff_map = []
    ratio_xyoff_map = []
    for logE in range(0,logE_nbins):
        data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
        fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
        ratio_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]

    #effective_matrix_rank = max(1,big_eigenvectors[0].shape[0])
    effective_matrix_rank = max(1,big_eigenvectors.shape[0])
    truth_params = [1e-3] * effective_matrix_rank
    fit_params = [1e-3] * effective_matrix_rank
    cr_qual = 0.
    sr_qual = 0.
    print (f'big_eigenvectors.shape = {big_eigenvectors.shape}') 

    print ('build big matrix...')

    big_on_matrix = []

    print (f'runlist = {runlist}')
    print (f'mimic_runlist = {mimic_runlist}')

    if 'MIMIC' in onoff:
        mimic_index = int(onoff.strip('MIMIC'))-1
        new_mimic_runlist = []
        for run in range(0,len(runlist)):
            if run>=len(mimic_runlist): continue
            if mimic_index>=len(mimic_runlist[run]): continue
            new_mimic_runlist += [mimic_runlist[run][mimic_index]]
        big_on_matrix, big_mask_matrix = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,new_mimic_runlist,max_runs=1e10,is_bkgd=False,is_on=True)
    elif 'ON' in onoff:
        big_on_matrix, big_mask_matrix = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,runlist,max_runs=1e10,is_bkgd=False,is_on=True)
    else:
        big_on_matrix, big_mask_matrix = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,runlist,max_runs=1e10,is_bkgd=False,is_on=False)

    #if big_on_matrix[0]==None or big_off_matrix[0]==None:
    #if big_on_matrix[0]==None:
    #if len(big_on_matrix)==0 or len(big_off_matrix)==0:
    if len(big_on_matrix)==0:
        print (f'No data. Break.')
        return [exposure_hours,avg_tel_elev,avg_tel_azim,truth_params,fit_params,sr_qual,cr_qual], incl_sky_map, data_sky_map, fit_sky_map, data_xyoff_map, fit_xyoff_map, ratio_xyoff_map

    for logE in range(0,logE_nbins):
        data_xyoff_map[logE].reset()
        fit_xyoff_map[logE].reset()
        ratio_xyoff_map[logE].reset()

    print ('fitting xyoff maps...')

    #for logE in range(0,logE_nbins):

    effective_matrix_rank = max(1,big_eigenvectors.shape[0])

    data_xyoff_map_1d = np.zeros_like(big_on_matrix[0])
    mask_xyoff_map_1d = np.zeros_like(big_mask_matrix[0])
    for entry in range(0,len(big_on_matrix)):
        data_xyoff_map_1d += np.array(big_on_matrix[entry])
        mask_xyoff_map_1d += np.array(big_mask_matrix[entry])

    template_norm = cosmic_ray_like_count(avg_xyoff_map_1d,region_type=0)
    on_data_norm = cosmic_ray_like_count(data_xyoff_map_1d,region_type=0)
    best_template_xyoff_map_1d = np.array(avg_xyoff_map_1d)*on_data_norm/template_norm

    diff_xyoff_map_1d = data_xyoff_map_1d - best_template_xyoff_map_1d
    fit_params = [0.] * effective_matrix_rank
    truth_params = big_eigenvectors @ diff_xyoff_map_1d

    init_params = [1e-4] * effective_matrix_rank
    stepsize = [1e-4] * effective_matrix_rank
    solution = minimize(
        cosmic_ray_like_chi2,
        x0=init_params,
        args=(fit_params,big_eigenvectors,diff_xyoff_map_1d,best_template_xyoff_map_1d,mask_xyoff_map_1d,0),
        #args=(init_params,big_eigenvectors,diff_xyoff_map_1d,best_template_xyoff_map_1d,-1),  # unblind
        method='L-BFGS-B',
        jac=None,
        options={'eps':stepsize,'ftol':0.0001},
    )
    fit_params = solution['x']
    #fit_params = truth_params

    fit_xyoff_map_1d = big_eigenvectors.T @ fit_params + best_template_xyoff_map_1d

    run_sr_chi2 = cosmic_ray_like_chi2(fit_params,big_eigenvalues,big_eigenvectors,diff_xyoff_map_1d,best_template_xyoff_map_1d,mask_xyoff_map_1d,1)
    run_cr_chi2 = cosmic_ray_like_chi2(fit_params,big_eigenvalues,big_eigenvectors,diff_xyoff_map_1d,best_template_xyoff_map_1d,mask_xyoff_map_1d,0)

    print ('===================================================================================')
    print (f'effective_matrix_rank = {effective_matrix_rank}')
    for entry in range(0,len(truth_params)):
        print (f'truth_params = {truth_params[entry]:0.1f}, fit_params = {fit_params[entry]:0.1f}')
    sum_truth_params = np.sum(truth_params)
    sum_fit_params = np.sum(fit_params)
    print (f'sum_truth_params = {sum_truth_params:0.1f}, sum_fit_params = {sum_fit_params:0.1f}')
    print (f'run_sr_chi2 = {run_sr_chi2:0.3f}, run_cr_chi2 = {run_cr_chi2:0.3f}')

    sr_data_count = cosmic_ray_like_count(data_xyoff_map_1d,region_type=1)
    fit_data_count = cosmic_ray_like_count(fit_xyoff_map_1d,region_type=1)
    init_data_count = cosmic_ray_like_count(best_template_xyoff_map_1d,region_type=1)
    cr_data_count = cosmic_ray_like_count(data_xyoff_map_1d,region_type=0)
    fit_cr_data_count = cosmic_ray_like_count(fit_xyoff_map_1d,region_type=0)

    print (f'sr_data_count = {sr_data_count:0.1f}, init_data_count = {init_data_count:0.1f}, fit_data_count = {fit_data_count:0.1f}')

    cr_qual = run_cr_chi2
    sr_qual = run_sr_chi2

    idx_1d = 0
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d += 1
                    data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += data_xyoff_map_1d[idx_1d-1]
                    fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += fit_xyoff_map_1d[idx_1d-1]

    print ('===================================================================================')
    for logE in range(0,logE_nbins):
        print (f'logE = {logE}')
        for gcut in range(0,gcut_bins):
            if gcut!=0: continue
            print (f'gcut = {gcut}')
            sum_data_xyoff_map = np.sum(data_xyoff_map[logE].waxis[:,:,gcut])
            sum_fit_xyoff_map = np.sum(fit_xyoff_map[logE].waxis[:,:,gcut])
            print (f'sum_data_xyoff_map = {sum_data_xyoff_map:0.1f}, sum_fit_xyoff_map = {sum_fit_xyoff_map:0.1f}')

    for logE in range(0,logE_nbins):
        for idx_x in range(0,xoff_bins[logE]):
            for idx_y in range(0,yoff_bins[logE]):
                sum_xyoff_map_cr = 0.
                sum_xyoff_map_sr = fit_xyoff_map[logE].waxis[idx_x,idx_y,0]
                for gcut in range(1,gcut_bins):
                    sum_xyoff_map_cr += fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
                ratio = 1.
                if sum_xyoff_map_cr>0.:
                    ratio = sum_xyoff_map_sr/sum_xyoff_map_cr
                ratio_xyoff_map[logE].waxis[idx_x,idx_y,0] = ratio

    for logE in range(0,logE_nbins):

        if xoff_bins[logE]==1: continue
        if yoff_bins[logE]==1: continue

        for idx_x in range(0,xoff_bins[logE]):
            for idx_y in range(0,yoff_bins[logE]):
                if abs(ratio_xyoff_map[logE].waxis[idx_x,idx_y,0])>100.:
                    print (f'Very large ratio!!! Reset to 1.')
                    ratio_xyoff_map[logE].waxis[idx_x,idx_y,0] = 1.


    #for logE in range(0,logE_nbins):

    #    if xoff_bins[logE]==1: continue
    #    if yoff_bins[logE]==1: continue

    #    avg_ratio = 0.
    #    count = 0.
    #    for idx_x in range(0,xoff_bins[logE]):
    #        for idx_y in range(0,yoff_bins[logE]):
    #            avg_ratio += ratio_xyoff_map[logE].waxis[idx_x,idx_y,0]
    #            count += 1.
    #    if count==0.: continue
    #    avg_ratio = avg_ratio/count

    #    rms_ratio = 0.
    #    count = 0.
    #    for idx_x in range(0,xoff_bins[logE]):
    #        for idx_y in range(0,yoff_bins[logE]):
    #            rms_ratio += pow(ratio_xyoff_map[logE].waxis[idx_x,idx_y,0]-avg_ratio,2)
    #            count += 1.
    #    if count==0.: continue
    #    rms_ratio = pow(rms_ratio/count,0.5)

    #    new_avg_ratio = 0.
    #    count = 0.
    #    for idx_x in range(0,xoff_bins[logE]):
    #        for idx_y in range(0,yoff_bins[logE]):
    #            if rms_ratio==0.: continue
    #            deviation = (ratio_xyoff_map[logE].waxis[idx_x,idx_y,0] - avg_ratio)/rms_ratio
    #            if abs(deviation)<2.:
    #                new_avg_ratio += ratio_xyoff_map[logE].waxis[idx_x,idx_y,0]
    #                count += 1.
    #    if count==0.: continue
    #    avg_ratio = new_avg_ratio/count

    #    new_rms_ratio = 0.
    #    count = 0.
    #    for idx_x in range(0,xoff_bins[logE]):
    #        for idx_y in range(0,yoff_bins[logE]):
    #            if rms_ratio==0.: continue
    #            deviation = (ratio_xyoff_map[logE].waxis[idx_x,idx_y,0] - avg_ratio)/rms_ratio
    #            if abs(deviation)<2.:
    #                new_rms_ratio += pow(ratio_xyoff_map[logE].waxis[idx_x,idx_y,0]-avg_ratio,2)
    #                count += 1.
    #    if count==0.: continue
    #    rms_ratio = pow(new_rms_ratio/count,0.5)

    #    for idx_x in range(0,xoff_bins[logE]):
    #        for idx_y in range(0,yoff_bins[logE]):
    #            if rms_ratio==0.: continue
    #            deviation = (ratio_xyoff_map[logE].waxis[idx_x,idx_y,0] - avg_ratio)/rms_ratio
    #            if abs(deviation)>3.:
    #                ratio_xyoff_map[logE].waxis[idx_x,idx_y,0] = avg_ratio

    #abort_job = False
    #for logE in range(0,logE_nbins):

    #    if xoff_bins[logE]==1: continue
    #    if yoff_bins[logE]==1: continue

    #    max_ratio = np.max(ratio_xyoff_map[logE].waxis[:,:,0])
    #    if max_ratio>100.:
    #        print (f'Very large ratio!!! Abort job.')
    #        abort_job = True

    #if abort_job:
    #    return [exposure_hours,avg_tel_elev,avg_tel_azim,truth_params,fit_params,sr_qual,cr_qual], incl_sky_map, data_sky_map, fit_sky_map, data_xyoff_map, fit_xyoff_map, ratio_xyoff_map


    run_count = 0
    for run in range(0,len(runlist)):

        run_number = runlist[run]
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

        if TelElevation<run_elev_cut: continue

        if 'MIMIC' in onoff:
            mimic_index = int(onoff.strip('MIMIC'))-1
            run_number = mimic_runlist[run][mimic_index]
            print (f'analyzing mimic run {run_number}')

            InputFile.Close()
            rootfile_name = f'{smi_input}/{run_number}.anasum.root'
            print (rootfile_name)
            if not os.path.exists(rootfile_name):
                print (f'file does not exist.')
                continue
            InputFile = ROOT.TFile(rootfile_name)

        TreeName = f'run_{run_number}/stereo/DL3EventTree'
        print (f'TreeName = {TreeName}')

        EvtTree = InputFile.Get(TreeName)
        total_entries = EvtTree.GetEntries()
        #print (f'total_entries = {total_entries}')
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
            MSCW = EvtTree.MSCW/MSCW_cut[logE]
            MSCL = EvtTree.MSCL/MSCL_cut[logE]
            GammaCut = EventGammaCut(MSCL,MSCW)

            Xsky = TelRAJ2000 + Xderot
            Ysky = TelDecJ2000 + Yderot

            if onoff=='OFF':
                found_gamma_source = CoincideWithBrightStars(Xsky, Ysky, gamma_source_coord)
                if found_gamma_source: continue

            incl_sky_map[logE].fill(Xsky,Ysky,0.5)
            if GammaCut>float(gcut_end): continue

            cr_correction = ratio_xyoff_map[logE].get_bin_content(Xoff,Yoff,0.5)
            #cr_correction = ratio_xyoff_map[logE].get_bin_content(Xderot,Yderot,0.5)
            if GammaCut<1.:
                cr_correction = 0.

            data_sky_map[logE].fill(Xsky,Ysky,GammaCut)
            fit_sky_map[logE].fill(Xsky,Ysky,0.5,weight=cr_correction)
    
        print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')

        InputFile.Close()
  
    tracemalloc.stop()
    if exposure_hours>0.:
        avg_tel_elev = avg_tel_elev/exposure_hours
        avg_tel_azim = avg_tel_azim/exposure_hours

    print (f'avg_tel_elev = {avg_tel_elev}')

    return [exposure_hours,avg_tel_elev,avg_tel_azim,truth_params,fit_params,sr_qual,cr_qual], incl_sky_map, data_sky_map, fit_sky_map, data_xyoff_map, fit_xyoff_map, ratio_xyoff_map


def cosmic_ray_like_chi2(try_params,ref_params,eigenvectors,diff_xyoff_map,init_xyoff_map,mask_xyoff_map,region_type):

    try_params = np.array(try_params)
    try_xyoff_map = eigenvectors.T @ try_params + init_xyoff_map

    sum_log_likelihood = 0.
    idx_1d = 0
    nbins = 0.
    n_expect_total = 0.
    n_data_total = 0.
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):

                    idx_1d += 1
                    diff = diff_xyoff_map[idx_1d-1]
                    init = init_xyoff_map[idx_1d-1]
                    mask = mask_xyoff_map[idx_1d-1]
                    weight = 1.

                    if region_type==0:
                        if gcut==0:
                            if mask>0.: # blind
                                diff = 0.
                                weight = fix_init_scale
                    elif region_type==1:
                        if gcut!=0: continue

                    n_expect = max(0.0001,try_xyoff_map[idx_1d-1])
                    n_data = diff + init

                    n_expect_total += n_expect
                    n_data_total += n_data
 
                    if use_poisson_likelihood:
                        if n_data==0.:
                            sum_log_likelihood += (n_expect)*weight
                        else:
                            sum_log_likelihood += (-1.*(n_data*np.log(n_expect) - n_expect - (n_data*np.log(n_data)-n_data)))*weight
                        nbins += 1.
                    else:
                        sum_log_likelihood += pow(n_expect-n_data,2)*weight

    if use_poisson_likelihood:
        sum_log_likelihood = sum_log_likelihood/nbins

    #sum_try_params = np.sum(try_params)
    #nuclear_norm_scale = 1.
    #if use_poisson_likelihood:
    #    sum_log_likelihood += nuclear_norm_scale*sum_try_params*sum_try_params/n_data_total
    #else:
    #    sum_log_likelihood += nuclear_norm_scale*sum_try_params*sum_try_params

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

    drawBrightStar = True
    drawPulsar = False
    drawSNR = False
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


def MakeSkymapCutout(skymap_input,cutout_frac):

    input_skymap_bins = len(skymap_input.xaxis)-1
    new_skymap_bins = int(cutout_frac*input_skymap_bins)
    xsky_start = skymap_input.xaxis[0]
    xsky_end = skymap_input.xaxis[len(skymap_input.xaxis)-1]
    ysky_start = skymap_input.yaxis[0]
    ysky_end = skymap_input.yaxis[len(skymap_input.yaxis)-1]
    xsky_center = 0.5*(xsky_start+xsky_end)
    ysky_center = 0.5*(ysky_start+ysky_end)
    new_xsky_start = xsky_center-0.5*(xsky_end-xsky_start)*cutout_frac
    new_xsky_end = xsky_center+0.5*(xsky_end-xsky_start)*cutout_frac
    new_ysky_start = ysky_center-0.5*(ysky_end-ysky_start)*cutout_frac
    new_ysky_end = ysky_center+0.5*(ysky_end-ysky_start)*cutout_frac
    #print (f'new_skymap_bins = {new_skymap_bins}')
    #print (f'xsky_start = {xsky_start}, xsky_end = {xsky_end}')
    #print (f'ysky_start = {ysky_start}, ysky_end = {ysky_end}')
    #print (f'new_xsky_start = {new_xsky_start}, new_xsky_end = {new_xsky_end}')
    #print (f'new_ysky_start = {new_ysky_start}, new_ysky_end = {new_ysky_end}')
    skymap_cutout = MyArray3D(x_bins=new_skymap_bins,start_x=new_xsky_start,end_x=new_xsky_end,y_bins=new_skymap_bins,start_y=new_ysky_start,end_y=new_ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)

    for idx_x in range(0,new_skymap_bins):
        for idx_y in range(0,new_skymap_bins):
            xsky = 0.5*(skymap_cutout.xaxis[idx_x]+skymap_cutout.xaxis[idx_x+1])
            ysky = 0.5*(skymap_cutout.yaxis[idx_y]+skymap_cutout.yaxis[idx_y+1])
            bin_content = skymap_input.get_bin_content(xsky,ysky,0.5)
            skymap_cutout.waxis[idx_x,idx_y,0] = bin_content

    return skymap_cutout

def PlotSkyMap(fig,label_z,logE_min,logE_max,hist_map_input,plotname,roi_x=[],roi_y=[],roi_r=[],max_z=0.,colormap='coolwarm',layer=0):

    E_min = pow(10.,logE_bins[logE_min])
    E_max = pow(10.,logE_bins[logE_max])

    hist_map = MakeSkymapCutout(hist_map_input,1.0)

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
    #for star in range(0,len(other_star_markers)):
    #    print (f'Star {other_star_labels[star]} RA = {other_star_markers[star][0]:0.1f}, Dec = {other_star_markers[star][1]:0.1f}')

    fig.clf()
    figsize_x = 7
    figsize_y = 7
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()

    label_x = 'RA [deg]'
    label_y = 'Dec [deg]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(hist_map.waxis[:,:,layer].T,origin='lower',extent=(xmax,xmin,ymin,ymax),aspect='auto',cmap=colormap)
    if max_z!=0.:
        im = axbig.imshow(hist_map.waxis[:,:,layer].T,origin='lower',extent=(xmax,xmin,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap=colormap)
    #cbar = fig.colorbar(im)

    divider = make_axes_locatable(axbig)
    cax = divider.append_axes("bottom", size="5%", pad=0.7)
    cbar = fig.colorbar(im,orientation="horizontal",cax=cax)
    cbar.set_label(label_z)

    font = {'family': 'serif', 'color':  'k', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

    favorite_color = 'k'
    if colormap=='magma':
        favorite_color = 'deepskyblue'
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
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='TeV':
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='Star':
            axbig.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c='k', marker='+', label=other_star_labels[star])
        txt = axbig.text(other_star_markers[star][0]-0.07, other_star_markers[star][1]+0.07, other_star_labels[star], fontdict=font, c=favorite_color)

    for roi in range(0,len(roi_x)):
        mycircle = plt.Circle( (roi_x[roi], roi_y[roi]), roi_r[roi], fill = False, color='white')
        axbig.add_patch(mycircle)

    lable_energy_range = f'E = {E_min:0.2f}-{E_max:0.2f} TeV'
    txt = axbig.text(xmax-0.14, ymax-0.21, lable_energy_range, fontdict=font)

    fig.savefig(f'output_plots/{plotname}.png',bbox_inches='tight')
    axbig.remove()

def GetFluxCalibration(energy):

    if doFluxCalibration:
        return 1.

    str_flux_calibration = ['2.45e+03', '6.99e+02', '7.64e+02', '1.18e+03', '1.17e+03', '2.64e+03', '6.20e+03', '1.52e+04', '3.71e+04']
    #str_flux_calibration = ['3.95e+02', '5.93e+02', '8.96e+02', '1.32e+03', '1.25e+03', '2.63e+03', '6.10e+03', '1.37e+04', '2.92e+04']

    flux_calibration = []
    for string in str_flux_calibration:
        flux_calibration.append(float(string))

    return 1./flux_calibration[energy]

def make_significance_map(data_sky_map,bkgd_sky_map,significance_sky_map,excess_sky_map):
  
    skymap_bins = len(data_sky_map.xaxis)-1

    for idx_x in range(0,skymap_bins):
        for idx_y in range(0,skymap_bins):
            data = data_sky_map.waxis[idx_x,idx_y,0]
            bkgd = bkgd_sky_map.waxis[idx_x,idx_y,0]
            data_err = pow(data,0.5)
            if data_err==0.: continue
            significance_sky_map.waxis[idx_x,idx_y,0] = (data-bkgd)/data_err
            excess_sky_map.waxis[idx_x,idx_y,0] = (data-bkgd)

def make_flux_map(incl_sky_map,data_sky_map,bkgd_sky_map,flux_sky_map,flux_err_sky_map,avg_energy,delta_energy):
  
    skymap_bins = len(data_sky_map.xaxis)-1

    norm_content_max = np.max(incl_sky_map.waxis[:,:,0])

    for idx_x in range(0,skymap_bins):
        for idx_y in range(0,skymap_bins):
            data = data_sky_map.waxis[idx_x,idx_y,0]
            norm = incl_sky_map.waxis[idx_x,idx_y,0]
            bkgd = bkgd_sky_map.waxis[idx_x,idx_y,0]
            if norm>0.:
                excess = data-bkgd
                error = pow(data,0.5)
                logE = logE_axis.get_bin(np.log10(avg_energy))
                correction = GetFluxCalibration(logE)/norm*pow(avg_energy,2)/(100.*100.*3600.)/delta_energy
                norm_ratio = norm/norm_content_max
                norm_weight = 1.
                if norm_ratio<0.3: norm_weight = 0.
                #norm_weight = 1./(1.+np.exp(-(norm_ratio-0.3)/0.05))
                flux = excess*correction*norm_weight
                flux_err = error*correction*norm_weight
                flux_sky_map.waxis[idx_x,idx_y,0] = flux
                flux_err_sky_map.waxis[idx_x,idx_y,0] = flux_err
            else:
                flux_sky_map.waxis[idx_x,idx_y,0] = 0.
                flux_err_sky_map.waxis[idx_x,idx_y,0] = 0.

def GetRadialProfile(hist_flux_skymap,hist_error_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,use_excl=True):

    deg2_to_sr =  3.046*1e-4
    pix_size = abs((hist_flux_skymap.yaxis[1]-hist_flux_skymap.yaxis[0])*(hist_flux_skymap.xaxis[1]-hist_flux_skymap.xaxis[0]))*deg2_to_sr
    bin_size = min(0.1,2.*(hist_flux_skymap.yaxis[1]-hist_flux_skymap.yaxis[0]))
    radial_axis = MyArray1D(x_nbins=int(roi_r/bin_size),start_x=0.,end_x=roi_r)

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
                if use_excl:
                    for roi in range(0,len(excl_roi_x)):
                        excl_distance = pow(pow(bin_ra-excl_roi_x[roi],2) + pow(bin_dec-excl_roi_y[roi],2),0.5)
                        if excl_distance<excl_roi_r[roi]: 
                            keep_event = False
                if keep_event:
                    pixel_array[br] += 1.*pix_size
                    brightness_array[br] += hist_flux_skymap.waxis[bx,by,0]
                    brightness_err_array[br] += pow(hist_error_skymap.waxis[bx,by,0],2)
        if pixel_array[br]==0.: continue
        brightness_array[br] = brightness_array[br]/pixel_array[br]
        brightness_err_array[br] = pow(brightness_err_array[br],0.5)/pixel_array[br]

    return radius_array, brightness_array, brightness_err_array

def GetRegionIntegral(hist_flux_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_error_skymap=None,use_excl=True):

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
            if use_excl:
                for roi in range(0,len(excl_roi_x)):
                    excl_distance = pow(pow(bin_ra-excl_roi_x[roi],2) + pow(bin_dec-excl_roi_y[roi],2),0.5)
                    if excl_distance<excl_roi_r[roi]: 
                        keep_event = False
            if keep_event:
                if not hist_flux_skymap.waxis[bx,by,0]==0.:
                    flux_sum += hist_flux_skymap.waxis[bx,by,0]
                    if hist_error_skymap==None:
                        flux_stat_err += hist_flux_skymap.waxis[bx,by,0]
                    else:
                        flux_stat_err += pow(hist_error_skymap.waxis[bx,by,0],2)
                else:
                    flux_sum += 0.
                    if hist_error_skymap==None:
                        flux_stat_err = max(hist_flux_skymap.waxis[bx,by,0],flux_stat_err)
                    else:
                        flux_stat_err = max(pow(hist_error_skymap.waxis[bx,by,0],2),flux_stat_err)
    flux_stat_err = pow(flux_stat_err,0.5)
    return flux_sum, flux_stat_err

def GetRegionSpectrum(hist_flux_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_error_skymap=None,use_excl=True):

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
        if hist_error_skymap==None:
            flux_sum, flux_stat_err = GetRegionIntegral(hist_flux_skymap[binE],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,use_excl=use_excl)
        else:
            flux_sum, flux_stat_err = GetRegionIntegral(hist_flux_skymap[binE],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_error_skymap=hist_error_skymap[binE],use_excl=use_excl)
        x_axis += [0.5*(pow(10.,logE_axis.xaxis[binE+1])+pow(10.,logE_axis.xaxis[binE]))]
        x_error += [0.5*(pow(10.,logE_axis.xaxis[binE+1])-pow(10.,logE_axis.xaxis[binE]))]
        y_axis += [flux_sum]
        y_error += [flux_stat_err]

    return x_axis, x_error, y_axis, y_error

def flux_crab_func(x):
    # TeV^{-1}cm^{-2}s^{-1}
    # Crab https://arxiv.org/pdf/1508.06442.pdf
    #return 37.5*pow(10,-12)*pow(x*1./1000.,-2.467-0.16*np.log(x/1000.))
    return 37.5*pow(10,-12)*pow(x,-2.467-0.16*np.log(x))

def GetHessSS433e():

    # https://indico.icc.ub.edu/event/46/contributions/1302/attachments/443/871/ID382_OliveraNieto_SS433.pdf

    energies = [0.045,0.354,0.660,0.969,1.276,1.583]
    fluxes = [-12.47,-12.57,-12.51,-12.53,-12.67,-12.70]
    flux_errs_up = [-12.31,-12.40,-12.40,-12.43,-12.54,-12.54]
    flux_errs_lo = [-12.69,-12.83,-12.66,-12.64,-12.83,-12.89]
    flux_errs = []

    erg_to_TeV = 0.62
    for entry in range(0,len(energies)):
        energies[entry] = pow(10.,energies[entry])
        fluxes[entry] = pow(10.,fluxes[entry])*erg_to_TeV
        flux_errs += [0.5*(pow(10.,flux_errs_up[entry])-pow(10.,flux_errs_lo[entry]))*erg_to_TeV]

    return energies, fluxes, flux_errs

def GetHessSS433w():

    # https://indico.icc.ub.edu/event/46/contributions/1302/attachments/443/871/ID382_OliveraNieto_SS433.pdf

    energies = [0.045,0.354,0.660,0.969,1.276,1.583]
    fluxes = [-12.36,-12.44,-12.74,-12.56,-12.93,-12.80]
    flux_errs_up = [-12.23,-12.30,-12.57,-12.46,-12.73,-12.61]
    flux_errs_lo = [-12.55,-12.63,-13.00,-12.68,-13.24,-13.07]
    flux_errs = []

    erg_to_TeV = 0.62
    for entry in range(0,len(energies)):
        energies[entry] = pow(10.,energies[entry])
        fluxes[entry] = pow(10.,fluxes[entry])*erg_to_TeV
        flux_errs += [0.5*(pow(10.,flux_errs_up[entry])-pow(10.,flux_errs_lo[entry]))*erg_to_TeV]

    return energies, fluxes, flux_errs


def PrintInformationRoI(fig,logE_min,logE_mid,logE_max,source_name,hist_data_skymap,hist_bkgd_skymap,hist_flux_skymap,hist_flux_err_skymap,hist_mimic_data_skymap,hist_mimic_bkgd_skymap,roi_name,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r):

    energy_axis, energy_error, flux, flux_stat_err = GetRegionSpectrum(hist_flux_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_error_skymap=hist_flux_err_skymap)
    energy_axis, energy_error, data, data_stat_err = GetRegionSpectrum(hist_data_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)
    energy_axis, energy_error, bkgd, bkgd_stat_err = GetRegionSpectrum(hist_bkgd_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)

    bkgd_syst_err = np.zeros_like(bkgd_stat_err)
    bkgd_incl_err = np.zeros_like(bkgd_stat_err)
    flux_syst_err = np.zeros_like(bkgd_stat_err)
    flux_incl_err = np.zeros_like(bkgd_stat_err)
    n_mimic = len(hist_mimic_data_skymap)
    list_mimic_data = []
    list_mimic_bkgd = []
    for mimic in range(0,n_mimic):
        mimic_energy_axis, mimic_energy_error, mimic_data, mimic_data_stat_err = GetRegionSpectrum(hist_mimic_data_skymap[mimic],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)
        mimic_energy_axis, mimic_energy_error, mimic_bkgd, mimic_bkgd_stat_err = GetRegionSpectrum(hist_mimic_bkgd_skymap[mimic],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)
        list_mimic_data += [mimic_data]
        list_mimic_bkgd += [mimic_bkgd]
    for binx in range(0,len(energy_axis)):
        stat_err = data_stat_err[binx]
        syst_err = 0.
        for mimic in range(0,n_mimic):
            syst_err += pow(list_mimic_data[mimic][binx]-list_mimic_bkgd[mimic][binx],2)
        if n_mimic>0:
            syst_err = pow(max(syst_err/float(n_mimic)-stat_err*stat_err,0.),0.5)
        bkgd_syst_err[binx] = syst_err
        bkgd_incl_err[binx] = pow(pow(stat_err,2)+pow(syst_err,2),0.5)
        if stat_err>0.:
            flux_syst_err[binx] = syst_err/stat_err*flux_stat_err[binx]
        else:
            flux_syst_err[binx] = 0.
        flux_incl_err[binx] = pow(pow(flux_syst_err[binx],2)+pow(flux_stat_err[binx],2),0.5)


    vectorize_f_crab = np.vectorize(flux_crab_func)
    ydata_crab_ref = pow(np.array(energy_axis),2)*vectorize_f_crab(energy_axis)

    flux_floor = []
    flux_cu = []
    flux_err_cu = []
    for binx in range(0,len(energy_axis)):
        if flux[binx]>0.:
            flux_floor += [flux[binx]]
            flux_cu += [flux[binx]/ydata_crab_ref[binx]]
            flux_err_cu += [flux_stat_err[binx]/ydata_crab_ref[binx]]
        else:
            flux_floor += [0.]
            flux_cu += [0.]
            flux_err_cu += [0.]

    print ('===============================================================================================================')
    formatted_numbers = ['%0.2e' % num for num in energy_axis]
    print ('energy_axis = %s'%(formatted_numbers))
    formatted_numbers = ['%0.2e' % num for num in flux_cu]
    print ('new flux_calibration = %s'%(formatted_numbers))

    print ('===============================================================================================================')
    print (f'RoI : {roi_name}')

    min_energy = pow(10.,logE_bins[logE_min])
    mid_energy = pow(10.,logE_bins[logE_mid])
    max_energy = pow(10.,logE_bins[logE_max])

    sum_data = 0.
    sum_bkgd = 0.
    sum_error = 0.
    for binx in range(0,len(energy_axis)):
        if energy_axis[binx]>min_energy and energy_axis[binx]<mid_energy:
            sum_data += data[binx]
            sum_bkgd += bkgd[binx]
            sum_error += bkgd_incl_err[binx]*bkgd_incl_err[binx]
    sum_error = pow(sum_error,0.5)
    significance = 0.
    if sum_error>0.:
        significance = (sum_data-sum_bkgd)/sum_error
    print (f'E = {min_energy:0.2f}-{mid_energy:0.2f} TeV, data = {sum_data:0.1f}, bkgd = {sum_bkgd:0.1f} +/- {sum_error:0.1f}, significance = {significance:0.1f} sigma')

    sum_data = 0.
    sum_bkgd = 0.
    sum_error = 0.
    for binx in range(0,len(energy_axis)):
        if energy_axis[binx]>mid_energy and energy_axis[binx]<max_energy:
            sum_data += data[binx]
            sum_bkgd += bkgd[binx]
            sum_error += bkgd_incl_err[binx]*bkgd_incl_err[binx]
    sum_error = pow(sum_error,0.5)
    significance = 0.
    if sum_error>0.:
        significance = (sum_data-sum_bkgd)/sum_error
    print (f'E = {mid_energy:0.2f}-{max_energy:0.2f} TeV, data = {sum_data:0.1f}, bkgd = {sum_bkgd:0.1f} +/- {sum_error:0.1f}, significance = {significance:0.1f} sigma')

    for binx in range(0,len(energy_axis)):
        significance = 0.
        if bkgd_incl_err[binx]>0.:
            significance = (data[binx]-bkgd[binx])/bkgd_incl_err[binx]
        print (f'E = {energy_axis[binx]:0.2f} TeV, data = {data[binx]:0.1f} +/- {data_stat_err[binx]:0.1f}, bkgd = {bkgd[binx]:0.1f} +/- {bkgd_syst_err[binx]:0.1f}, flux = {flux[binx]:0.2e} +/- {flux_incl_err[binx]:0.2e} TeV/cm2/s ({flux_cu[binx]:0.2f} CU), significance = {significance:0.1f} sigma')
    print ('===============================================================================================================')

    fig.clf()
    figsize_x = 7
    figsize_y = 5
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'Energy [TeV]'
    label_y = 'Flux in C.U.'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.set_xscale('log')
    axbig.set_yscale('log')
    axbig.errorbar(energy_axis,flux_cu,flux_err_cu,xerr=energy_error,color='k',marker='_',ls='none',zorder=1)
    fig.savefig(f'output_plots/{source_name}_roi_flux_crab_unit_{roi_name}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    figsize_x = 7
    figsize_y = 5
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'Energy [TeV]'
    label_y = 'Flux [TeV/cm2/s]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.set_xscale('log')
    axbig.set_yscale('log')
    axbig.fill_between(energy_axis,np.array(flux_floor)-np.array(flux_incl_err),np.array(flux_floor)+np.array(flux_incl_err),alpha=0.2,color='b',zorder=0)
    axbig.errorbar(energy_axis,flux,flux_stat_err,xerr=energy_error,color='k',marker='_',ls='none',label=f'VERITAS ({roi_name})',zorder=1)
    if 'SS433' in source_name:
        HessSS433e_energies, HessSS433e_fluxes, HessSS433e_flux_errs = GetHessSS433e()
        HessSS433w_energies, HessSS433w_fluxes, HessSS433w_flux_errs = GetHessSS433w()
        axbig.errorbar(HessSS433e_energies,HessSS433e_fluxes,HessSS433e_flux_errs,marker='s',ls='none',label='HESS eastern',zorder=2)
        axbig.errorbar(HessSS433w_energies,HessSS433w_fluxes,HessSS433w_flux_errs,marker='s',ls='none',label='HESS western',zorder=3)
    axbig.legend(loc='best')
    fig.savefig(f'output_plots/{source_name}_roi_energy_flux_{roi_name}.png',bbox_inches='tight')
    axbig.remove()


def DefineRegionOfInterest(src_name,src_ra,src_dec):

    region_x = []
    region_y = []
    region_r = []
    region_name = []

    if 'Crab' in src_name:

        region_x = [src_ra]
        region_y = [src_dec]
        region_r = [calibration_radius]
        region_name = ['center']

        region_x += [84.4]
        region_y += [21.1]
        region_r += [calibration_radius]
        region_name += ['star']

    elif 'Geminga' in src_name:

        region_x = [src_ra]
        region_y = [src_dec]
        region_r = [1.5]
        region_name = ['center']

    elif 'SNR_G189_p03' in src_name:

        src_x = 94.25
        src_y = 22.57
        region_x += [src_x]
        region_y += [src_y]
        region_r += [1.0]
        region_name += ['center']

    elif 'PSR_J2021_p4026' in src_name:

        src_x = 305.21
        src_y = 40.43
        region_x += [src_x]
        region_y += [src_y]
        region_r += [1.2]
        region_name += ['SNR']

    elif 'PSR_J1907_p0602' in src_name:

        region_x = [287.05]
        region_y = [6.39]
        region_r = [1.2]
        region_name = ['3HWC']

        region_x += [288.0833333]
        region_y += [4.9166667]
        region_r += [0.2]
        region_name += ['SS443']

    elif 'PSR_J1856_p0245' in src_name:

        region_x = [284.3]
        region_y = [2.7]
        region_r = [1.0]
        region_name = ['J1857+026']

        region_x += [284.6]
        region_y += [2.1]
        region_r += [0.2]
        region_name += ['J1858+020']

    elif 'SS433' in src_name:
    
        #SS 433 SNR
        #region_x += [288.0833333]
        #region_y += [4.9166667]
        #region_r += [0.2]
        #region_name += ['SNR']
    
        #SS 433 e1
        region_x += [288.404]
        region_y += [4.930]
        region_r += [0.2]
        region_name += ['SS433e1']
        #region_x = [288.35,288.50,288.65,288.8]
        #region_y = [4.93,4.92,4.93,4.94]
        #region_r = [0.1,0.1,0.1,0.1]
    
        #SS 433 w1
        region_x += [287.654]
        region_y += [5.037]
        region_r += [0.2]
        region_name += ['SS433w1']

    else:

        region_x = [src_ra]
        region_y = [src_dec]
        region_r = [3.0]
        region_name = ['center']


    return region_name, region_x, region_y, region_r

def SaveFITS(skymap_input,filename):

    image_length_x = len(skymap_input.xaxis)-1
    image_length_y = len(skymap_input.yaxis)-1
    array_shape = (image_length_y,image_length_x)
    new_image_data = np.zeros(array_shape)

    ref_pix = 1
    central_ra = skymap_input.xaxis[ref_pix]
    central_dec = skymap_input.yaxis[ref_pix]
    pixel_size = skymap_input.yaxis[1]-skymap_input.yaxis[0]
    reduced_wcs = wcs.WCS(naxis=2)
    reduced_wcs.wcs.ctype = ['RA---TAN','DEC--TAN']
    reduced_wcs.wcs.crpix = [ref_pix,ref_pix] # Set the reference pixel coordinates
    reduced_wcs.wcs.crval = [central_ra, central_dec] # Set the reference values for the physical coordinates
    reduced_wcs.wcs.cdelt = [pixel_size,pixel_size]

    for pix_x in range(0,new_image_data.shape[0]):
        for pix_y in range(0,new_image_data.shape[1]):
            new_image_data[pix_x,pix_y] += skymap_input.waxis[pix_x,pix_y,0]

    fits.writeto('output_plots/%s.fits'%(filename), new_image_data, reduced_wcs.to_header(), overwrite=True)

# Our function to fit is going to be a sum of two-dimensional Gaussians
def gaussian(x, y, x0, y0, sigma, A):
    #return A * np.exp( -((x-x0)/(2.*sigma))**2 -((y-y0)/(2.*sigma))**2)
    #return A * np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma*sigma))/(2*np.pi*sigma*sigma)
    return A * np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma*sigma))
# https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//4):
       arr += gaussian(x, y, *args[i*4:i*4+4])
    return arr

def fit_2d_model(data_sky_map,bkgd_sky_map, src_x, src_y):

    nbins_x = len(data_sky_map.xaxis)-1
    nbins_y = len(data_sky_map.yaxis)-1
    lon_min = data_sky_map.xaxis[0]
    lon_max = data_sky_map.xaxis[len(data_sky_map.xaxis)-1]
    lat_min = data_sky_map.yaxis[0]
    lat_max = data_sky_map.yaxis[len(data_sky_map.yaxis)-1]
    x_axis = np.linspace(lon_min,lon_max,nbins_x)
    y_axis = np.linspace(lat_min,lat_max,nbins_y)
    X_grid, Y_grid = np.meshgrid(x_axis, y_axis)
    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    XY_stack = np.vstack((X_grid.ravel(), Y_grid.ravel()))

    image_excess = np.zeros((nbins_x,nbins_y))
    image_error = np.zeros((nbins_x,nbins_y))
    for binx in range (0,nbins_x):
        for biny in range (0,nbins_y):
            image_excess[biny,binx] = data_sky_map.waxis[binx,biny,0] - bkgd_sky_map.waxis[binx,biny,0]
            image_error[biny,binx] = max(pow(data_sky_map.waxis[binx,biny,0],0.5),1.)

    #print ('set initial avlues and bounds')
    initial_prms = []
    bound_upper_prms = []
    bound_lower_prms = []
    lon = src_x
    lat = src_y
    sigma = 0.03807
    initial_prms += [(lon,lat,sigma,10.)]
    centroid_range = 0.5
    bound_lower_prms += [(lon-centroid_range,lat-centroid_range,sigma+0.0,0.)]
    bound_upper_prms += [(lon+centroid_range,lat+centroid_range,sigma+2.0,1e10)]
    # Flatten the initial guess parameter list.
    p0 = [p for prms in initial_prms for p in prms]
    p0_lower = [p for prms in bound_lower_prms for p in prms]
    p0_upper = [p for prms in bound_upper_prms for p in prms]
    print ('p0 = %s'%(p0))

    popt, pcov = curve_fit(_gaussian, XY_stack, image_excess.ravel(), p0, sigma=image_error.ravel(), absolute_sigma=True, bounds=(p0_lower,p0_upper))
    fit_src_x = popt[0*4+0]
    fit_src_x_err = pow(pcov[0*4+0][0*4+0],0.5)
    print ('fit_src_x = %0.3f +/- %0.3f'%(fit_src_x,fit_src_x_err))
    fit_src_y = popt[0*4+1]
    fit_src_y_err = pow(pcov[0*4+1][0*4+1],0.5)
    print ('fit_src_y = %0.3f +/- %0.3f'%(fit_src_y,fit_src_y_err))
    fit_src_sigma = popt[0*4+2]
    fit_src_sigma_err = pow(pcov[0*4+2][0*4+2],0.5)
    print ('fit_src_sigma = %0.3f +/- %0.3f'%(fit_src_sigma,fit_src_sigma_err))
    fit_src_A = popt[0*4+3]
    print ('fit_src_A = %0.1e'%(fit_src_A))

    distance_to_psr = pow(pow(fit_src_x-src_x,2)+pow(fit_src_y-src_y,2),0.5)
    distance_to_psr_err = pow(pow(fit_src_x_err,2)+pow(fit_src_y_err,2),0.5)
    print ('distance_to_psr = %0.3f +/- %0.3f'%(distance_to_psr,fit_src_x_err))

    profile_fit = _gaussian(XY_stack, *popt)
    residual = image_excess.ravel() - profile_fit
    chisq = np.sum((residual/image_error.ravel())**2)
    dof = len(image_excess.ravel())-4
    print ('chisq/dof = %0.3f'%(chisq/dof))

def plot_radial_profile_with_systematics(fig,plotname,flux_sky_map,flux_err_sky_map,mimic_flux_sky_map,mimic_flux_err_sky_map,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r):

    on_radial_axis, on_profile_axis, on_profile_err_axis = GetRadialProfile(flux_sky_map,flux_err_sky_map,roi_x,roi_y,2.0,excl_roi_x,excl_roi_y,excl_roi_r)
    all_radial_axis, all_profile_axis, all_profile_err_axis = GetRadialProfile(flux_sky_map,flux_err_sky_map,roi_x,roi_y,2.0,excl_roi_x,excl_roi_y,excl_roi_r,use_excl=False)
    n_mimic = len(mimic_flux_sky_map)
    fig.clf()
    figsize_x = 7
    figsize_y = 5
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'angular distance [deg]'
    label_y = 'surface brightness [$\mathrm{TeV}\ \mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{sr}^{-1}$]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    mimic_profile_axis = []
    mimic_profile_err_axis = []
    for mimic in range(0,n_mimic):
        radial_axis, profile_axis, profile_err_axis = GetRadialProfile(mimic_flux_sky_map[mimic],mimic_flux_err_sky_map[mimic],roi_x,roi_y,2.0,excl_roi_x,excl_roi_y,excl_roi_r,use_excl=False)
        axbig.errorbar(radial_axis,profile_axis,profile_err_axis,marker='+',ls='none',zorder=mimic+1)
        mimic_profile_axis += [profile_axis]
        mimic_profile_err_axis += [profile_err_axis]
    profile_syst_err_axis = []
    for binx in range(0,len(on_profile_axis)):
        syst_err = 0.
        stat_err = 0.
        for mimic in range(0,n_mimic):
            syst_err += pow(mimic_profile_axis[mimic][binx],2)
            stat_err += pow(mimic_profile_err_axis[mimic][binx],2)
        syst_err = max(0.,syst_err-stat_err)
        if n_mimic>0:
            syst_err = pow(syst_err/float(n_mimic),0.5)
        profile_syst_err_axis += [pow(pow(syst_err,2)+pow(on_profile_err_axis[binx],2),0.5)]
    baseline_yaxis = [0. for i in range(0,len(on_radial_axis))]
    axbig.plot(on_radial_axis, baseline_yaxis, color='b', ls='dashed')
    axbig.fill_between(on_radial_axis,-np.array(profile_syst_err_axis),np.array(profile_syst_err_axis),alpha=0.2,color='b',zorder=0)
    fig.savefig(f'output_plots/{plotname}_mimic.png',bbox_inches='tight')
    axbig.remove()

    baseline_yaxis = [0. for i in range(0,len(on_radial_axis))]
    fig.clf()
    figsize_x = 7
    figsize_y = 5
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'angular distance [deg]'
    label_y = 'surface brightness [$\mathrm{TeV}\ \mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{sr}^{-1}$]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(on_radial_axis, baseline_yaxis, color='b', ls='dashed')
    axbig.errorbar(all_radial_axis,all_profile_axis,all_profile_err_axis,color='r',marker='+',ls='none',zorder=1)
    axbig.errorbar(on_radial_axis,on_profile_axis,on_profile_err_axis,color='k',marker='+',ls='none',zorder=2)
    axbig.fill_between(on_radial_axis,np.array(on_profile_axis)-np.array(profile_syst_err_axis),np.array(on_profile_axis)+np.array(profile_syst_err_axis),alpha=0.2,color='b',zorder=0)
    fig.savefig(f'output_plots/{plotname}.png',bbox_inches='tight')
    axbig.remove()


def build_radial_symmetric_model(radial_symmetry_sky_map,on_radial_axis,on_profile_axis,roi_x,roi_y):

    deg2_to_sr =  3.046*1e-4
    pix_size = abs((radial_symmetry_sky_map.yaxis[1]-radial_symmetry_sky_map.yaxis[0])*(radial_symmetry_sky_map.xaxis[1]-radial_symmetry_sky_map.xaxis[0]))*deg2_to_sr
    radial_bin_size = on_radial_axis[1]-on_radial_axis[0]
    for bx in range(0,len(radial_symmetry_sky_map.xaxis)-1):
        for by in range(0,len(radial_symmetry_sky_map.yaxis)-1):
            for br in range(0,len(on_radial_axis)):
                bin_ra = 0.5*(radial_symmetry_sky_map.xaxis[bx]+radial_symmetry_sky_map.xaxis[bx+1])
                bin_dec = 0.5*(radial_symmetry_sky_map.yaxis[by]+radial_symmetry_sky_map.yaxis[by+1])
                distance = pow(pow(bin_ra-roi_x,2) + pow(bin_dec-roi_y,2),0.5)
                if distance<on_radial_axis[br]-0.5*radial_bin_size: continue
                if distance>=on_radial_axis[br]+0.5*radial_bin_size: continue
                if on_profile_axis[br]<0.: continue
                radial_symmetry_sky_map.waxis[bx,by,0] = on_profile_axis[br]*pix_size
                break


