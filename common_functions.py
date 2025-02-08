
import os, sys
import math
import random
import ROOT
import numpy as np
import pickle
import csv
from scipy.optimize import least_squares, minimize
from scipy.optimize import curve_fit
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import NullFormatter
import tracemalloc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import wcs
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as astropy_unit
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.table import Table

cr_tag = os.environ.get("CR_TAG")
bin_tag = os.environ.get("BIN_TAG")
norm_tag = os.environ.get("NORM_TAG")
eigen_tag = os.environ.get("EIGEN_TAG")
sky_tag = os.environ.get("SKY_TAG")
smi_output = os.environ.get("SMI_OUTPUT")

run_elev_cut = 25.

#min_NImages = 2
min_NImages = 3
max_Roff = 1.7
max_EmissionHeight_cut = 20.
min_EmissionHeight_cut = 6.
max_MeanPedvar_cut = 11.
min_MeanPedvar_cut = 3.
max_Rcore = 400.
min_Rcore = 0.
min_Energy_cut = 0.02
max_Energy_cut = 100.0
MVA_cut = 0.5

xoff_start = -2.
xoff_end = 2.
yoff_start = -2.
yoff_end = 2.

#logE_bins = [-1.00,-0.90,-0.80,-0.70,-0.60,-0.50,-0.40,-0.25,0.00,0.25,0.50,0.75,1.00,1.25] # logE TeV
logE_bins = [-0.90,-0.80,-0.70,-0.60,-0.50,-0.40,-0.25,0.00,0.25,0.50,0.75,1.00,1.25] # logE TeV
#logE_bins = [-0.80,-0.70,-0.60,-0.50,-0.40,-0.25,0.00,0.25,0.50,0.75,1.00,1.25] # logE TeV
#logE_bins = [-0.60,-0.50,-0.40,-0.25,0.00,0.25,0.50,0.75,1.00,1.25] # logE TeV
logE_nbins = len(logE_bins)-1

#MSCW_cut = [0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60]
#MSCL_cut = [0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70]
#str_flux_calibration = ['1.80e+02', '3.40e+02', '6.90e+02', '1.42e+03', '2.51e+03', '2.85e+03', '2.53e+03', '3.20e+03', '8.51e+03', '2.51e+04', '1.14e+05', '3.70e+05', '1.09e+06']
MSCW_cut = [0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60]
MSCL_cut = [0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70]
str_flux_calibration = ['3.40e+02', '6.90e+02', '1.42e+03', '2.51e+03', '2.85e+03', '2.53e+03', '3.20e+03', '8.51e+03', '2.51e+04', '1.14e+05', '3.70e+05', '1.09e+06']
#MSCW_cut = [0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60]
#MSCL_cut = [0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70]
#str_flux_calibration = ['6.90e+02', '1.42e+03', '2.51e+03', '2.85e+03', '2.53e+03', '3.20e+03', '8.51e+03', '2.51e+04', '1.14e+05', '3.70e+05', '1.09e+06']
#MSCW_cut = [0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60]
#MSCL_cut = [0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70,0.70]
#str_flux_calibration = ['2.51e+03', '2.85e+03', '2.53e+03', '3.20e+03', '8.51e+03', '2.51e+04', '1.14e+05', '3.70e+05', '1.09e+06']

skymap_size = 3.
skymap_bins = 20
#fine_skymap_bins = 20
#skymap_bins = 60
fine_skymap_bins = 120

#doFluxCalibration = True
doFluxCalibration = False
calibration_radius = 0.15 # need to be larger than the PSF and smaller than the integration radius

#coordinate_type = 'galactic'
coordinate_type = 'icrs'

#logE_threshold = -99
#logE_threshold = 0
logE_threshold = 1
#logE_threshold = 2
fov_mask_radius = 10.
gcut_bins = 3
matrix_rank = 1
matrix_rank_fullspec = 16
xyoff_map_nbins = 9
xyvar_map_nbins = 20
use_poisson_likelihood = True
use_init = False
use_fft = False
use_mono = False

if eigen_tag=='init':
    matrix_rank_fullspec = 1
    use_init = True

if 'fullspec' in eigen_tag:
    matrix_rank_fullspec = int(eigen_tag.strip('fullspec'))
elif 'monospec' in eigen_tag:
    matrix_rank_fullspec = int(eigen_tag.strip('monospec'))
    use_mono = True

if 'nbin' in bin_tag:
    xyoff_map_nbins = int(bin_tag.strip('nbin'))

if 'cr' in cr_tag:
    gcut_bins = int(cr_tag.strip('cr'))

if 'free' in norm_tag:
    fov_mask_radius = 10.
elif 'fov' in norm_tag:
    fov_mask_radius = float(norm_tag.strip('fov'))/10.

gcut_start = 0
gcut_end = gcut_bins
gcut_weight = [1.] * gcut_bins

Normalized_MSCL_cut = []
Normalized_MSCW_cut = []
if gcut_bins==6:
    Normalized_MSCL_cut += [1.]
    Normalized_MSCL_cut += [3.]
    Normalized_MSCW_cut += [1.]
    Normalized_MSCW_cut += [3.]
    Normalized_MSCW_cut += [5.]
elif gcut_bins==8:
    #Normalized_MSCL_cut += [1.]
    #Normalized_MSCL_cut += [2.]
    #Normalized_MSCW_cut += [1.]
    #Normalized_MSCW_cut += [2.]
    #Normalized_MSCW_cut += [3.]
    #Normalized_MSCW_cut += [4.]
    Normalized_MSCL_cut += [1.]
    Normalized_MSCL_cut += [3.]
    Normalized_MSCW_cut += [1.]
    Normalized_MSCW_cut += [3.]
    Normalized_MSCW_cut += [5.]
    Normalized_MSCW_cut += [7.]

xoff_bins = [xyoff_map_nbins for logE in range(0,logE_nbins)]
yoff_bins = [xyoff_map_nbins for logE in range(0,logE_nbins)]
xvar_bins = [xyvar_map_nbins for logE in range(0,logE_nbins)]
yvar_bins = [xyvar_map_nbins for logE in range(0,logE_nbins)]

smi_aux = os.environ.get("SMI_AUX")
smi_dir = os.environ.get("SMI_DIR")


def weighted_least_square_solution(mtx_input,vec_output,vec_weight,plot_tag=''):

    x = np.array(mtx_input)
    y = np.array(vec_output)
    #w = np.diag(np.array(vec_weight))
    print (f"x.shape = {x.shape}")
    print (f"y.shape = {y.shape}")
    #print (f"w.shape = {w.shape}")

    over_constrained = True
    if x.shape[0]<=x.shape[1]:
        over_constrained = False
        print ("system is not over-constrained.")

    #xTwx = x.T @ w @ x
    xTwx = x.T @ x
    U, S, VT = np.linalg.svd(xTwx, full_matrices=False)

    fig, ax = plt.subplots()
    figsize_x = 6
    figsize_y = 4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = 'order $i$'
    label_y = 'singular value $S_{i}$'
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.plot(S)
    ax.set_yscale('log')
    fig.savefig(f'output_plots/least_square_sigularvalue_{plot_tag}.png',bbox_inches='tight')
    del fig
    del ax
    plt.close()

    S_pseudo_inv = np.diag(1 / S)
    for entry in range(0,len(S)):
        if S[entry]/S[0]<pow(10.,-4.0):
            S_pseudo_inv[entry][entry] = 0.

    inv_xTwx = VT.T @ S_pseudo_inv @ U.T

    #A = inv_xTwx @ x.T @ w @ y
    A = inv_xTwx @ x.T @ y
    y_predict = x @ A
    y_err = np.sqrt(np.square(y - y_predict))
    #A_err = inv_xTwx @ x.T @ w @ y_err
    A_err = inv_xTwx @ x.T @ y_err

    if over_constrained:
        return A, A_err
    else:
        return A, 2.*A

def significance_li_and_ma(N_on, N_bkg, N_bkg_err):

    if (N_on+N_bkg)<=1.0:
        return 0.

    sign = 1.
    if N_on<N_bkg:
        sign = -1.

    # in the limit of alpha = 1.

    on_nlogn = 0.
    if N_on>0.:
        on_nlogn = N_on*np.log(2.*(N_on/(N_on+N_bkg)))
    bkg_nlogn = 0.
    if N_bkg>0.:
        bkg_nlogn = N_bkg*np.log(2.*(N_bkg/(N_on+N_bkg)))

    chi_square_stat = 2 * (on_nlogn + bkg_nlogn)
    chi_square = chi_square_stat

    if N_bkg_err>0.:
        chi_square_syst = pow((N_on-N_bkg)/N_bkg_err,2)
        chi_square = 1. / (1./chi_square_stat + 1./chi_square_syst)

    S = sign * pow(chi_square,0.5)

    return S

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

        paired_mimic_runs = []
        if os.path.exists(input_mimic_file):
            mimic_inputFile = open(input_mimic_file)
            for mimic_line in mimic_inputFile:
                line_split = mimic_line.split()
                on_run = int(line_split[0])
                mimic_run = int(line_split[1])
                if on_run==int(on_line):
                    paired_mimic_runs += [mimic_run]

        all_runlist += [(onrun_elev,int(on_line),paired_off_runs,paired_mimic_runs)]

    all_runlist.sort(key=sortFirst,reverse=True) # from high-elev to low-elev
    #all_runlist.sort(key=sortFirst) # from low-elev to high-elev
    #random.shuffle(all_runlist)

    for run in range(0,len(all_runlist)):
        on_runlist += [all_runlist[run][1]]
        off_runlist += [all_runlist[run][2]]
        mimic_runlist += [all_runlist[run][3]]

    return on_runlist, off_runlist, mimic_runlist

def ReadOffRunListFromFile(smi_input,input_onlist_file, input_offlist_file, mimic_index):

    on_runlist = []
    on_runlist_elev = []

    print (f'onlist_file = {input_onlist_file}')
    print (f'offlist_file = {input_offlist_file}')

    onlist_file = open(input_onlist_file)
    offlist_file = open(input_offlist_file)

    if mimic_index==0:
        for line in onlist_file:
            line_split = line.split()
            on_runlist += [int(line_split[0])]
    else:
        last_on_runnumber = 0
        last_mimic_index = 0
        for line in onlist_file:
            line_split = line.split()
            on_runnumber = int(line_split[0])
            off_runnumber = int(line_split[1])
            if on_runnumber!=last_on_runnumber:
                last_on_runnumber = on_runnumber
                last_mimic_index = 0
            last_mimic_index += 1
            if last_mimic_index==mimic_index:
                on_runlist += [off_runnumber]

    for run in range(0,len(on_runlist)):
        onrun_elev, onrun_azim = GetRunElevAzim(smi_input,on_runlist[run])
        on_runlist_elev += [(onrun_elev,on_runlist[run])]

    #on_runlist_elev.sort(key=sortFirst,reverse=True)
    #on_runlist_elev.sort(key=sortFirst)
    random.shuffle(on_runlist_elev)

    runs_per_batch = 1
    on_runlist_sorted = []
    small_runlist = []
    run_count = 0
    for run in range(0,len(on_runlist_elev)):
        run_elev = on_runlist_elev[run][0]
        run_number = on_runlist_elev[run][1]
        small_runlist += [run_number]
        run_count += 1
        if run_count==runs_per_batch or run_count==len(on_runlist_elev)-1:
            on_runlist_sorted += [small_runlist]
            small_runlist = []
            run_count = 0
        if len(on_runlist_elev)-1-run<=runs_per_batch:
            runs_per_batch = 1e10

    off_runs_per_batch = 1000
    off_runlist = [[] for i in range(0,len(on_runlist_sorted))]
    off_run_count = [0 for i in range(0,len(on_runlist_sorted))]
    for line in offlist_file:
        line_split = line.split()
        on_runnumber = int(line_split[0])
        off_runnumber = int(line_split[1])
        for batch in range(0,len(on_runlist_sorted)):
            if off_run_count[batch]==off_runs_per_batch:
                continue
            for run in range(0,len(on_runlist_sorted[batch])):
                if on_runnumber==on_runlist_sorted[batch][run]:
                    off_runlist[batch] += [off_runnumber]
                    off_run_count[batch] += 1

    print (f'on_runlist = {on_runlist}')
    print (f'off_runlist = {off_runlist}')

    return off_runlist

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

    def __del__(self):
        #print('Destructor called, MyArray3D deleted.')
        pass

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
        for idx_x in range(0,len(self.xaxis)-1):
            if abs(self.xaxis[idx_x]-value_x)<=abs(self.delta_x[idx_x]) and abs(self.xaxis[idx_x+1]-value_x)<abs(self.delta_x[idx_x]):
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
        if distance>3.: continue
        #print (f'{line_split}')
        bright_stars_coord += [[star_ra,star_dec]]
    #print (f'Found {len(bright_stars_coord)} Gamma-ray sources.')
    return bright_stars_coord

def GetRunTimecuts(input_runnumber):

    list_timecuts = []
    inputFile = open(f'{smi_aux}/timecuts_allruns.txt')
    for line in inputFile:
        line_split = line.split()
        runnumber = int(line_split[0])
        if runnumber!=input_runnumber: continue
        if line_split[1]=='None': continue
        timecuts = line_split[1].split(',')
        for cut in range(0,len(timecuts)):
            timecut_start = float(timecuts[cut].split('/')[0])
            timecut_end = float(timecuts[cut].split('/')[1])
            list_timecuts += [[timecut_start,timecut_end]]
    return list_timecuts

def ApplyTimeCuts(evt_time, list_timecuts):

    pass_cut = True
    for cut in range(0,len(list_timecuts)):
        timecut_start = list_timecuts[cut][0]
        timecut_end = list_timecuts[cut][1]
        if evt_time>timecut_start and evt_time<timecut_end:
            pass_cut = False
    return pass_cut

def CalculateExposure(start_time, end_time, list_timecuts):

    total_time = end_time - start_time
    removed_time = 0.
    for cut in range(0,len(list_timecuts)):
        timecut_start = list_timecuts[cut][0]
        timecut_end = list_timecuts[cut][1]
        removed_time += (timecut_end-timecut_start)

    return total_time - removed_time

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

    if gcut_bins==6:
        if abs(MSCL)<Normalized_MSCL_cut[0] and abs(MSCW)<Normalized_MSCW_cut[0]:
            GammaCut = 0.5
        elif abs(MSCL)<Normalized_MSCL_cut[0] and abs(MSCW)<Normalized_MSCW_cut[1]:
            GammaCut = 1.5
        elif abs(MSCL)<Normalized_MSCL_cut[0] and abs(MSCW)<Normalized_MSCW_cut[2]:
            GammaCut = 2.5
        elif abs(MSCL)<Normalized_MSCL_cut[1] and abs(MSCW)<Normalized_MSCW_cut[0]:
            GammaCut = 3.5
        elif abs(MSCL)<Normalized_MSCL_cut[1] and abs(MSCW)<Normalized_MSCW_cut[1]:
            GammaCut = 4.5
        elif abs(MSCL)<Normalized_MSCL_cut[1] and abs(MSCW)<Normalized_MSCW_cut[2]:
            GammaCut = 5.5
    elif gcut_bins==8:
        if abs(MSCL)<Normalized_MSCL_cut[0] and abs(MSCW)<Normalized_MSCW_cut[0]:
            GammaCut = 0.5
        elif abs(MSCL)<Normalized_MSCL_cut[0] and abs(MSCW)<Normalized_MSCW_cut[1]:
            GammaCut = 1.5
        elif abs(MSCL)<Normalized_MSCL_cut[0] and abs(MSCW)<Normalized_MSCW_cut[2]:
            GammaCut = 2.5
        elif abs(MSCL)<Normalized_MSCL_cut[0] and abs(MSCW)<Normalized_MSCW_cut[3]:
            GammaCut = 3.5
        elif abs(MSCL)<Normalized_MSCL_cut[1] and abs(MSCW)<Normalized_MSCW_cut[0]:
            GammaCut = 4.5
        elif abs(MSCL)<Normalized_MSCL_cut[1] and abs(MSCW)<Normalized_MSCW_cut[1]:
            GammaCut = 5.5
        elif abs(MSCL)<Normalized_MSCL_cut[1] and abs(MSCW)<Normalized_MSCW_cut[2]:
            GammaCut = 6.5
        elif abs(MSCL)<Normalized_MSCL_cut[1] and abs(MSCW)<Normalized_MSCW_cut[3]:
            GammaCut = 7.5


    return GammaCut

def convert_multivar_map3d_to_vector1d(xyoff_map, xyvar_map):

    xyoff_map_1d = []
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    xyoff_map_1d += [xyoff_map[logE].waxis[idx_x,idx_y,gcut]]
    for logE in range(0,logE_nbins):
        for idx_x in range(0,xvar_bins[logE]):
            for idx_y in range(0,yvar_bins[logE]):
                xyoff_map_1d += [xyvar_map[logE].waxis[idx_x,idx_y,0]]

    return xyoff_map_1d

def convert_multivar_vector1d_to_map3d(multivar_map_1d):

    xyoff_map = []
    xyvar_map = []
    for logE in range(0,logE_nbins):
        xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
        end_x = Normalized_MSCL_cut[len(Normalized_MSCL_cut)-1]
        end_y = Normalized_MSCW_cut[len(Normalized_MSCW_cut)-1]
        xyvar_map += [MyArray3D(x_bins=xvar_bins[logE],start_x=-1.,end_x=end_x,y_bins=yvar_bins[logE],start_y=-1.,end_y=end_y,z_bins=1,start_z=0.,end_z=1.)]

    idx_1d = 0
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d += 1
                    xyoff_map[logE].waxis[idx_x,idx_y,gcut] = multivar_map_1d[idx_1d-1]
    for logE in range(0,logE_nbins):
        for idx_x in range(0,xvar_bins[logE]):
            for idx_y in range(0,yvar_bins[logE]):
                idx_1d += 1
                xyvar_map[logE].waxis[idx_x,idx_y,0] = multivar_map_1d[idx_1d-1]

    return xyoff_map, xyvar_map

def convert_multivar_to_xyvar_vector1d(multivar_map_1d):

    xyoff_map, xyvar_map =  convert_multivar_vector1d_to_map3d(multivar_map_1d)

    xyvar_map_1d = []
    for logE in range(0,logE_nbins):
        for idx_x in range(0,xvar_bins[logE]):
            for idx_y in range(0,yvar_bins[logE]):
                xyvar_map_1d += [xyvar_map[logE].waxis[idx_x,idx_y,0]]

    return np.array(xyvar_map_1d)

def find_index_for_xyvar_vector1d():

    idx_1d = 0
    idx_1d_output = []
    for logE in range(0,logE_nbins):
        idx_1d_x = []
        for idx_x in range(0,xvar_bins[logE]):
            idx_1d_y = []
            for idx_y in range(0,yvar_bins[logE]):
                idx_1d_y += [idx_1d]
                idx_1d += 1
            idx_1d_x += [idx_1d_y]
        idx_1d_output += [idx_1d_x]

    return idx_1d_output

def convert_multivar_to_xyoff_vector1d(multivar_map_1d):

    xyoff_map, xyvar_map =  convert_multivar_vector1d_to_map3d(multivar_map_1d)

    xyoff_map_1d = []
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    xyoff_map_1d += [xyoff_map[logE].waxis[idx_x,idx_y,gcut]]

    return np.array(xyoff_map_1d)

def find_index_for_xyoff_vector1d():

    idx_1d = 0
    idx_1d_output = []
    for gcut in range(0,gcut_bins):
        idx_1d_logE = []
        for logE in range(0,logE_nbins):
            idx_1d_x = []
            for idx_x in range(0,xoff_bins[logE]):
                idx_1d_y = []
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d_y += [idx_1d]
                    idx_1d += 1
                idx_1d_x += [idx_1d_y]
            idx_1d_logE += [idx_1d_x]
        idx_1d_output += [idx_1d_logE]

    return idx_1d_output

def convert_xyoff_vector1d_to_map3d(xyoff_map_1d):

    xyoff_map = []
    for logE in range(0,logE_nbins):
        xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

    xyoff_idx_1d = find_index_for_xyoff_vector1d()
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d = xyoff_idx_1d[gcut][logE][idx_x][idx_y]
                    xyoff_map[logE].waxis[idx_x,idx_y,gcut] = xyoff_map_1d[idx_1d]

    return xyoff_map

def build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,runlist,max_runs=1e10,is_bkgd=True,specific_run=0):

    big_matrix_fullspec = []
    big_mask_matrix_fullspec = []

    big_exposure_time = []
    big_elevation = 0.

    region_name = source_name
    roi_name,roi_ra,roi_dec,roi_r = DefineRegionOfMask(region_name,src_ra,src_dec)

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
    
        xyvar_map = []
        xyvar_mask_map = []
        xyoff_map = []
        xyoff_mask_map = []
        for logE in range(0,logE_nbins):
            end_x = Normalized_MSCL_cut[len(Normalized_MSCL_cut)-1]
            end_y = Normalized_MSCW_cut[len(Normalized_MSCW_cut)-1]
            xyvar_map += [MyArray3D(x_bins=xvar_bins[logE],start_x=-1.,end_x=end_x,y_bins=yvar_bins[logE],start_y=-1.,end_y=end_y,z_bins=1,start_z=0.,end_z=1.)]
            xyvar_mask_map += [MyArray3D(x_bins=xvar_bins[logE],start_x=-1.,end_x=end_x,y_bins=yvar_bins[logE],start_y=-1.,end_y=end_y,z_bins=1,start_z=0.,end_z=1.)]
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

        list_timecuts = GetRunTimecuts(int(run_number))
        print (f"run_number = {run_number}, list_timecuts = {list_timecuts}")

        TreeName = f'run_{run_number}/stereo/DL3EventTree'
        EvtTree = InputFile.Get(TreeName)
        total_entries = EvtTree.GetEntries()
        #print (f'total_entries = {total_entries}')
        EvtTree.GetEntry(0)
        Time0 = EvtTree.timeOfDay
        EvtTree.GetEntry(total_entries-1)
        Time1 = EvtTree.timeOfDay
        exposure = CalculateExposure(Time0, Time1, list_timecuts)
        big_exposure_time += [exposure]
        big_elevation += TelElevation * exposure
        for entry in range(0,total_entries):
            EvtTree.GetEntry(entry)
            RA = EvtTree.RA
            DEC = EvtTree.DEC
            Xoff = EvtTree.Xoff
            Yoff = EvtTree.Yoff
            Xderot = EvtTree.Xderot
            Yderot = EvtTree.Yderot
            Energy = EvtTree.Energy
            NImages = EvtTree.NImages
            EmissionHeight = EvtTree.EmissionHeight
            MeanPedvar = EvtTree.MeanPedvar
            Xcore = EvtTree.XCore
            Ycore = EvtTree.YCore
            Time = EvtTree.timeOfDay
            Roff = pow(Xoff*Xoff+Yoff*Yoff,0.5)
            Rcore = pow(Xcore*Xcore+Ycore*Ycore,0.5)
            logE = logE_axis.get_bin(np.log10(Energy))

            if NImages<2: continue
            if not ApplyTimeCuts(Time-Time0,list_timecuts): continue
            if logE<0: continue
            if logE>=len(xyoff_map): continue
            if Energy<min_Energy_cut: continue
            if Energy>max_Energy_cut: continue

            MSCW = EvtTree.MSCW/MSCW_cut[logE]
            MSCL = EvtTree.MSCL/MSCL_cut[logE]
            GammaCut = EventGammaCut(MSCL,MSCW)
            if GammaCut>float(gcut_end): continue

            if NImages<min_NImages: continue
            if EmissionHeight>max_EmissionHeight_cut: continue
            if EmissionHeight<min_EmissionHeight_cut: continue
            if MeanPedvar>max_MeanPedvar_cut: continue
            if MeanPedvar<min_MeanPedvar_cut: continue
            if Roff>max_Roff: continue
            if Rcore>max_Rcore: continue
            if Rcore<min_Rcore: continue

            Xsky = TelRAJ2000 + Xderot
            Ysky = TelDecJ2000 + Yderot

            found_bright_star = CoincideWithBrightStars(Xsky, Ysky, bright_star_coord)
            found_gamma_source = CoincideWithBrightStars(Xsky, Ysky, gamma_source_coord)

            #mirror_Xsky = TelRAJ2000 - Xderot
            #mirror_Ysky = TelDecJ2000 - Yderot
            #found_mirror_star = CoincideWithBrightStars(mirror_Xsky, mirror_Ysky, bright_star_coord)
            #found_mirror_gamma_source = CoincideWithBrightStars(mirror_Xsky, mirror_Ysky, gamma_source_coord)

            if is_bkgd:
                if found_bright_star:
                    xyoff_mask_map[logE].fill(Xoff,Yoff,0.5)
                if found_gamma_source:
                    xyoff_mask_map[logE].fill(Xoff,Yoff,0.5)
            else:
                found_roi = CoincideWithRegionOfInterest(Xsky, Ysky, roi_ra, roi_dec, roi_r)
                if found_roi:
                    xyoff_mask_map[logE].fill(Xoff,Yoff,GammaCut)


            xyoff_map[logE].fill(Xoff,Yoff,GammaCut)

            xyvar_map[logE].fill(MSCL,MSCW,0.5)
            if GammaCut<1.:
                xyvar_mask_map[logE].fill(MSCL,MSCW,0.5)


        if is_bkgd:
            for logE in range(0,logE_nbins):
                for gcut in range(0,1):
                    for idx_x in range(0,xoff_bins[logE]):
                        for idx_y in range(0,yoff_bins[logE]):
                            if xyoff_mask_map[logE].waxis[idx_x,idx_y,gcut]>0.:
                                bin_coord = xyoff_mask_map[logE].get_bin_center(idx_x,idx_y,gcut)
                                bin_idx = xyoff_mask_map[logE].get_bin(-1.*bin_coord[0],-1.*bin_coord[1],bin_coord[2])
                                xyoff_map[logE].waxis[idx_x,idx_y,gcut] = xyoff_map[logE].waxis[bin_idx[0],bin_idx[1],bin_idx[2]]
    
        xyoff_map_1d = convert_multivar_map3d_to_vector1d(xyoff_map, xyvar_map)
        xyoff_mask_map_1d = convert_multivar_map3d_to_vector1d(xyoff_mask_map, xyvar_mask_map)
        big_matrix_fullspec += [xyoff_map_1d]
        big_mask_matrix_fullspec += [xyoff_mask_map_1d]

        InputFile.Close()
        if run_count==max_runs: break

    big_elevation = big_elevation / np.sum(np.array(big_exposure_time))
    print (f'batch elevation = {big_elevation}')

    return big_elevation, big_exposure_time, big_matrix_fullspec, big_mask_matrix_fullspec

def prepare_vector_for_least_square(multivar_map_1d):

    xyoff_map, xyvar_map =  convert_multivar_vector1d_to_map3d(multivar_map_1d)

    sr_map_1d = [0. for logE in range(0,logE_nbins)]
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    if gcut==0:
                        sr_map_1d[logE] += xyoff_map[logE].waxis[idx_x,idx_y,gcut]

    cr_map_1d = []
    for logE in range(0,logE_nbins):
        for gcut in range(1,gcut_bins):
            cr_map_1d += [np.sum(xyoff_map[logE].waxis[:,:,gcut])]

    return np.array(sr_map_1d), np.array(cr_map_1d)

def residual_correction_fullspec(
        logE,
        data_xyoff_map,
        bkgd_xyoff_map,
    ):

    #residual = []
    #for gcut in range(0,gcut_bins):
    #    data_sum = np.sum(data_xyoff_map.waxis[:,:,gcut])
    #    bkgd_sum = np.sum(bkgd_xyoff_map.waxis[:,:,gcut])
    #    if bkgd_sum>0.:
    #        residual += [(data_sum-bkgd_sum)]
    #    else:
    #        residual += [0.]
    #if residual[3]==0.:
    #    residual[0] = 0.
    #else:
    #    residual[0] = residual[1] * residual[2] / residual[3]


    norm = []
    residual = []
    for gcut in range(0,gcut_bins):
        data_sum = np.sum(data_xyoff_map.waxis[:,:,gcut])
        bkgd_sum = np.sum(bkgd_xyoff_map.waxis[:,:,gcut])
        residual += [(data_sum-bkgd_sum)]
        norm += [bkgd_sum]

    for idx_x in range(0,xoff_bins[logE]):
        for idx_y in range(0,yoff_bins[logE]):
            if norm[gcut_bins-1]==0.: continue
            old_bkgd = bkgd_xyoff_map.waxis[idx_x,idx_y,0]
            bkgd_xyoff_map.waxis[idx_x,idx_y,0] += 1. * residual[gcut_bins-1]/norm[gcut_bins-1] * old_bkgd



def cosmic_ray_like_chi2_fullspec(
        try_params,
        eigenvectors,
        data_xyoff_map,
        mask_xyoff_map,
        init_xyoff_map,
        syst_xyoff_map,
        norm_constraint,
    ):

    rng = np.random.default_rng()

    sum_log_likelihood = 0.
    lsq_log_likelihood = 0.

    try_params = np.array(try_params)
    try_xyoff_map = eigenvectors.T @ try_params

    xyoff_idx_1d = find_index_for_xyoff_vector1d()

    #for entry in range(0,len(try_params)-1):
    #    if abs(try_params[entry+1])>abs(try_params[entry]):
    #        lsq_log_likelihood += (abs(try_params[entry+1])-abs(try_params[entry]))/abs(try_params[entry])

    if norm_constraint:
        for logE in range(0,logE_nbins):
            for gcut in range(0,gcut_bins):

                weight_gcut = gcut_weight[gcut]
                n_expect_gcut = 0.
                n_syst_gcut = 0.
                n_data_gcut = 0.

                for idx_x in range(0,xoff_bins[logE]):
                    for idx_y in range(0,yoff_bins[logE]):

                        idx_1d = xyoff_idx_1d[gcut][logE][idx_x][idx_y]
                        data = data_xyoff_map[idx_1d]
                        init = init_xyoff_map[idx_1d]
                        syst = syst_xyoff_map[idx_1d]

                        n_expect = max(0.0001,try_xyoff_map[idx_1d])
                        n_expect_gcut += n_expect

                        n_data = data
                        if gcut==0:
                            n_data = init
                        n_data_gcut += n_data
                       
                        n_syst = 0.
                        if gcut==0:
                            n_syst = syst
                        n_syst_gcut += n_syst

                if gcut==0 or n_data_gcut==0.:
                    weight_gcut = 0.
                else:
                    weight_gcut = 1.

                #log_likelihood = significance_li_and_ma(n_data_gcut, n_expect_gcut, n_syst_gcut)
                log_likelihood = significance_li_and_ma(n_data_gcut, n_expect_gcut, 0.)
                lsq_log_likelihood += pow(log_likelihood,2) * weight_gcut
                #if n_data_gcut==0.:
                #    lsq_log_likelihood += n_expect_gcut*weight_gcut
                #else:
                #    lsq_log_likelihood += (-1.*(n_data_gcut*np.log(n_expect_gcut) - n_expect_gcut - (n_data_gcut*np.log(n_data_gcut)-n_data_gcut)))*weight_gcut


    sum_weight = 0.
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):

                    idx_1d = xyoff_idx_1d[gcut][logE][idx_x][idx_y]
                    data = data_xyoff_map[idx_1d]
                    init = init_xyoff_map[idx_1d]
                    mask = mask_xyoff_map[idx_1d]

                    n_expect = max(0.0001,try_xyoff_map[idx_1d])
                    n_data = data

                    weight = gcut_weight[gcut]

                    if gcut==0:
                        if mask>0.:
                            weight = 0.

                    sum_weight += weight

                    if use_poisson_likelihood:
                        log_likelihood = significance_li_and_ma(n_data, n_expect, 0.)
                        sum_log_likelihood += pow(log_likelihood,2) * weight
                        #log_likelihood = 0.
                        #if n_data==0.:
                        #    log_likelihood = (n_expect)
                        #    sum_log_likelihood += log_likelihood * weight
                        #else:
                        #    log_likelihood = -1. * (n_data*np.log(n_expect) - n_expect - (n_data*np.log(n_data)-n_data))
                        #    sum_log_likelihood += log_likelihood * weight
                    else:
                        sum_log_likelihood += pow(n_expect-n_data,2) * weight


    return sum_log_likelihood + lsq_log_likelihood
    #return sum_log_likelihood



def cosmic_ray_like_count_fullspec(xyoff_map,region_type=0):

    count = 0.
    idx_1d = 0
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d += 1

                    weight = 1.

                    count += xyoff_map[idx_1d-1]*weight

    return count

def cosmic_ray_like_count(logE,xyoff_map,region_type=0):

    count = 0.
    idx_1d = 0
    for gcut in range(0,gcut_bins):
        for idx_x in range(0,xoff_bins[logE]):
            for idx_y in range(0,yoff_bins[logE]):
                idx_1d += 1
                if region_type==-1:
                    if gcut==0: continue
                else:
                    if gcut!=region_type: continue

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

def ReadFermiHighEnergyCatalog():

    source_name = []
    source_ra = []
    source_dec = []

    filename = 'gll_psch_v07.fit'
    hdu_list = fits.open(filename)

    # point sources
    table_index = 1 
    mytable = Table.read(filename, hdu=table_index)
    for entry in range(0,len(mytable)):
        Source_Name = mytable[entry]['Source_Name']
        RAJ2000 = mytable[entry]['RAJ2000']
        DEJ2000 = mytable[entry]['DEJ2000']
        source_name += [Source_Name]
        source_ra += [RAJ2000]
        source_dec += [DEJ2000]

    # extended sources
    table_index = 2 
    mytable = Table.read(filename, hdu=table_index)
    for entry in range(0,len(mytable)):
        Source_Name = mytable[entry]['Source_Name']
        RAJ2000 = mytable[entry]['RAJ2000']
        DEJ2000 = mytable[entry]['DEJ2000']
        source_name += [Source_Name]
        source_ra += [RAJ2000]
        source_dec += [DEJ2000]

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
    target_ts = ''
    for line in inputFile:
        if line.split(' ')[0]=='<source':
            for block in range(0,len(line.split(' '))):
                if 'Unc_' in line.split(' ')[block]: continue
                if 'TS_value=' in line.split(' ')[block]:
                    target_ts = line.split('TS_value="')[1].split('"')[0]
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
            keep_source = True
            if target_ra=='': 
                keep_source = False
            #if target_type=='PointSource': 
            #    keep_source = False
            if target_ts=='': 
                keep_source = False
            elif float(target_ts)<100.: 
                keep_source = False
            if not keep_source: 
                target_ts = ''
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
    drawPulsar = False
    drawSNR = False
    drawLHAASO = False
    drawFermi = False
    drawHAWC = True
    drawTeV = False

    if drawBrightStar:
        star_name, star_ra, star_dec = ReadBrightStarListFromFile()
        for src in range(0,len(star_name)):
            src_ra = star_ra[src]
            src_dec = star_dec[src]
            other_stars += [star_name[src]]
            other_stars_type += ['Star']
            other_star_coord += [[src_ra,src_dec,0.]]

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
        #fermi_name, fermi_ra, fermi_dec = ReadFermiCatelog()
        fermi_name, fermi_ra, fermi_dec = ReadFermiHighEnergyCatalog()
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

def PlotCountProjection(fig,label_z,logE_min,logE_max,hist_map_data,hist_map_bkgd,label_x,label_y,plotname,hist_map_syst=None,roi_x=[],roi_y=[],roi_r=[],max_z=0.,colormap='coolwarm',layer=0):

    E_min = pow(10.,logE_bins[logE_min])
    E_max = pow(10.,logE_bins[logE_max])

    hist_map = MakeSkymapCutout(hist_map_data,1.0)

    xmin = hist_map.xaxis.min()
    xmax = hist_map.xaxis.max()
    ymin = hist_map.yaxis.min()
    ymax = hist_map.yaxis.max()

    other_stars, other_star_type, other_star_coord = GetGammaSourceInfo() 

    if coordinate_type == 'galactic':
        for star in range(0,len(other_stars)):
            other_star_coord[star][0], other_star_coord[star][1] = ConvertRaDecToGalactic(other_star_coord[star][0], other_star_coord[star][1])

    other_star_labels = []
    other_star_types = []
    other_star_markers = []
    star_range = 0.8*(xmax-xmin)/2.
    source_ra = (xmax+xmin)/2.
    source_dec = (ymax+ymin)/2.
    n_stars = 0
    for star in range(0,len(other_stars)):
        if abs(source_ra-other_star_coord[star][0])>star_range: continue
        if abs(source_dec-other_star_coord[star][1])>star_range: continue
        other_star_markers += [[other_star_coord[star][0],other_star_coord[star][1],other_star_coord[star][2]]]
        other_star_labels += ['%s'%(other_stars[star])]
        other_star_types += [other_star_type[star]]
        n_stars += 1

    hist_map_significance = MyArray3D()
    hist_map_significance.just_like(hist_map_data)
    hist_map_excess = MyArray3D()
    hist_map_excess.just_like(hist_map_data)
    make_significance_map(hist_map_data,hist_map_bkgd,hist_map_significance,hist_map_excess,syst_sky_map=hist_map_syst)

    x_pix_size = max(0.1,abs(hist_map_data.xaxis[1]-hist_map_data.xaxis[0]))
    y_pix_size = max(0.1,abs(hist_map_data.yaxis[1]-hist_map_data.yaxis[0]))
    x_proj_axis = MyArray1D(x_nbins=round(abs(xmax-xmin)/x_pix_size),start_x=xmin,end_x=xmax)
    y_proj_axis = MyArray1D(x_nbins=round(abs(ymax-ymin)/y_pix_size),start_x=ymax,end_x=ymin)
    y_proj_axis_inv = MyArray1D(x_nbins=round(abs(ymax-ymin)/y_pix_size),start_x=ymin,end_x=ymax)

    x_axis_array = []
    x_count_array = []
    x_bkgd_array = []
    x_syst_array = []
    x_error_array = []
    for br in range(0,len(x_proj_axis.xaxis)-1):
        x_axis_array += [0.5*(x_proj_axis.xaxis[br]+x_proj_axis.xaxis[br+1])]
        x_count_array += [0.]
        x_bkgd_array += [0.]
        x_syst_array += [0.]
        x_error_array += [0.]

    y_axis_array = []
    y_count_array = []
    y_bkgd_array = []
    y_syst_array = []
    y_error_array = []
    for br in range(0,len(y_proj_axis.xaxis)-1):
        y_axis_array += [0.5*(y_proj_axis.xaxis[br]+y_proj_axis.xaxis[br+1])]
        y_count_array += [0.]
        y_bkgd_array += [0.]
        y_syst_array += [0.]
        y_error_array += [0.]

    for bx in range(0,len(hist_map_data.xaxis)-2):
        for by in range(0,len(hist_map_data.yaxis)-2):

            bin_ra = 0.5*(hist_map_data.xaxis[bx]+hist_map_data.xaxis[bx+1])
            bin_dec = 0.5*(hist_map_data.yaxis[by]+hist_map_data.yaxis[by+1])

            for br in range(0,len(x_proj_axis.xaxis)-1):
                keep_event = False
                if abs(bin_ra-x_proj_axis.xaxis[br])<=x_pix_size and abs(bin_ra-x_proj_axis.xaxis[br+1])<=x_pix_size: 
                    keep_event = True
                if keep_event:
                    x_count_array[br] += hist_map_data.waxis[bx,by,0]
                    x_bkgd_array[br] += hist_map_bkgd.waxis[bx,by,0]
                    x_syst_array[br] += hist_map_syst.waxis[bx,by,0]

            for br in range(0,len(y_proj_axis.xaxis)-1):
                keep_event = False
                if abs(bin_dec-y_proj_axis.xaxis[br])<=y_pix_size and abs(bin_dec-y_proj_axis.xaxis[br+1])<=y_pix_size: 
                    keep_event = True
                if keep_event:
                    y_count_array[br] += hist_map_data.waxis[bx,by,0]
                    y_bkgd_array[br] += hist_map_bkgd.waxis[bx,by,0]
                    y_syst_array[br] += hist_map_syst.waxis[bx,by,0]

    for br in range(0,len(x_proj_axis.xaxis)-1):
        x_error_array[br] = pow(x_count_array[br],0.5)
        x_syst_array[br] = pow(x_syst_array[br],0.5)

    for br in range(0,len(y_proj_axis.xaxis)-1):
        y_error_array[br] = pow(y_count_array[br],0.5)
        y_syst_array[br] = pow(y_syst_array[br],0.5)

    # Define the locations for the axes
    left, width = 0.12, 0.6
    bottom, height = 0.12, 0.6
    bottom_h = bottom+height+0.03
    left_h = left+width+0.03
     
    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.20] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.20, height] # dimensions of y-histogram

    # Set up the size of the figure
    #fig = plt.figure(1, figsize=(9.5,9))
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(8)

    fig.clf()
    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx = plt.axes(rect_histx) # x histogram
    axHisty = plt.axes(rect_histy) # y histogram

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Plot the temperature data
    cax = (axTemperature.imshow(hist_map_data.waxis[:,:,layer].T,extent=(xmax,xmin,ymin,ymax),aspect='auto',origin='lower',cmap=colormap))

    for roi in range(0,len(roi_x)):
        mycircle = plt.Circle( (roi_x[roi], roi_y[roi]), roi_r[roi], fill = False, color='black')
        axTemperature.add_patch(mycircle)

    #Plot the axes labels
    axTemperature.set_xlabel(label_x)
    axTemperature.set_ylabel(label_y)

    #Plot the histograms
    axHistx.plot(x_axis_array,x_bkgd_array,color='r',ls='solid',label='Background model')
    axHistx.errorbar(x_axis_array,x_bkgd_array,yerr=x_syst_array,color='r',marker='.',ls='none')
    axHisty.plot(y_bkgd_array,y_axis_array,color='r',ls='solid')
    axHisty.errorbar(y_bkgd_array,y_axis_array,xerr=y_syst_array,color='r',marker='.',ls='none')
    axHistx.errorbar(x_axis_array,x_count_array,yerr=x_error_array,color='k',marker='.',ls='none',label='Observation data')
    axHisty.errorbar(y_count_array,y_axis_array,xerr=y_error_array,color='k',marker='.',ls='none')
    axHistx.set_xlim(axHistx.get_xlim()[::-1])
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    axHistx.yaxis.set_major_formatter(formatter)
    axHistx.legend(loc='best')

    font = {'family': 'serif', 'color':  'white', 'weight': 'normal', 'size': 10, 'rotation': 0.,}
    lable_energy_range = f'E = {E_min:0.2f}-{E_max:0.2f} TeV'
    txt = axTemperature.text(xmax-0.14, ymax-0.21, lable_energy_range, fontdict=font)

    fig.savefig(f'output_plots/count_{plotname}.png',bbox_inches='tight')


    # Set up the geometry of the three plots
    rect_temperature = [left, bottom - 0.12, width, height + 0.12] # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.20] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.20, height] # dimensions of y-histogram

    fig.clf()
    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx = plt.axes(rect_histx) # x histogram
    axHisty = plt.axes(rect_histy) # y histogram

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Plot the temperature data
    cax = axTemperature.imshow(hist_map_significance.waxis[:,:,layer].T,extent=(xmax,xmin,ymin,ymax),vmin=-3.,vmax=3.,aspect='auto',origin='lower',cmap='coolwarm')

    for roi in range(0,len(roi_x)):
        mycircle = plt.Circle( (roi_x[roi], roi_y[roi]), roi_r[roi], fill = False, color='black')
        axTemperature.add_patch(mycircle)

    divider = make_axes_locatable(axTemperature)
    cax_app = divider.append_axes("bottom", size="5%", pad=0.7)
    cbar = fig.colorbar(cax,orientation="horizontal",cax=cax_app)
    cbar.set_label('significance')

    favorite_color = 'k'
    font = {'family': 'serif', 'color':  favorite_color, 'weight': 'normal', 'size': 10, 'rotation': 0.,}

    for star in range(0,len(other_star_markers)):
        marker_size = 60
        if other_star_types[star]=='PSR':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=1.5*marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='SNR':
            mycircle = plt.Circle( (other_star_markers[star][0], other_star_markers[star][1]), other_star_markers[star][2], fill = False, color=favorite_color)
            axTemperature.add_patch(mycircle)
        if other_star_types[star]=='LHAASO':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='HAWC':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='Fermi':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='MSP':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='TeV':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='Star':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c='k', marker='+', label=other_star_labels[star])
        txt = axTemperature.text(other_star_markers[star][0]-0.07, other_star_markers[star][1]+0.07, other_star_labels[star], fontdict=font, c=favorite_color)


    #Plot the axes labels
    axTemperature.set_xlabel(label_x)
    axTemperature.set_ylabel(label_y)

    #Plot the histograms
    axHistx.plot(x_axis_array,x_bkgd_array,color='r',ls='solid',label='Background model')
    axHistx.errorbar(x_axis_array,x_bkgd_array,yerr=x_syst_array,color='r',marker='.',ls='none')
    axHisty.plot(y_bkgd_array,y_axis_array,color='r',ls='solid')
    axHisty.errorbar(y_bkgd_array,y_axis_array,xerr=y_syst_array,color='r',marker='.',ls='none')
    axHistx.errorbar(x_axis_array,x_count_array,yerr=x_error_array,color='k',marker='.',ls='none',label='Observation data')
    axHisty.errorbar(y_count_array,y_axis_array,xerr=y_error_array,color='k',marker='.',ls='none')
    axHistx.set_xlim(axHistx.get_xlim()[::-1])
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    axHistx.yaxis.set_major_formatter(formatter)
    axHistx.legend(loc='best')

    font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 10, 'rotation': 0.,}
    lable_energy_range = f'E = {E_min:0.2f}-{E_max:0.2f} TeV'
    txt = axTemperature.text(xmax-0.14, ymax-0.21, lable_energy_range, fontdict=font)

    fig.savefig(f'output_plots/significance_{plotname}.png',bbox_inches='tight')


    fig.clf()
    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx = plt.axes(rect_histx) # x histogram
    axHisty = plt.axes(rect_histy) # y histogram

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Plot the temperature data
    cax = axTemperature.imshow(hist_map_excess.waxis[:,:,layer].T,extent=(xmax,xmin,ymin,ymax),aspect='auto',origin='lower',cmap=colormap)

    for roi in range(0,len(roi_x)):
        mycircle = plt.Circle( (roi_x[roi], roi_y[roi]), roi_r[roi], fill = False, color='black')
        axTemperature.add_patch(mycircle)

    divider = make_axes_locatable(axTemperature)
    cax_app = divider.append_axes("bottom", size="5%", pad=0.7)
    cbar = fig.colorbar(cax,orientation="horizontal",cax=cax_app)
    cbar.set_label('excess count')

    favorite_color = 'k'
    if colormap=='magma':
        favorite_color = 'deepskyblue'
    font = {'family': 'serif', 'color':  favorite_color, 'weight': 'normal', 'size': 10, 'rotation': 0.,}

    for star in range(0,len(other_star_markers)):
        marker_size = 60
        if other_star_types[star]=='PSR':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=1.5*marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='SNR':
            mycircle = plt.Circle( (other_star_markers[star][0], other_star_markers[star][1]), other_star_markers[star][2], fill = False, color=favorite_color)
            axTemperature.add_patch(mycircle)
        if other_star_types[star]=='LHAASO':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='HAWC':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='Fermi':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='MSP':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='TeV':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c=favorite_color, marker='+', label=other_star_labels[star])
        if other_star_types[star]=='Star':
            axTemperature.scatter(other_star_markers[star][0], other_star_markers[star][1], s=marker_size, c='k', marker='+', label=other_star_labels[star])
        txt = axTemperature.text(other_star_markers[star][0]-0.07, other_star_markers[star][1]+0.07, other_star_labels[star], fontdict=font, c=favorite_color)

    #Plot the axes labels
    axTemperature.set_xlabel(label_x)
    axTemperature.set_ylabel(label_y)

    #Plot the histograms
    axHistx.plot(x_axis_array,x_bkgd_array,color='r',ls='solid',label='Background model')
    axHistx.errorbar(x_axis_array,x_bkgd_array,yerr=x_syst_array,color='r',marker='.',ls='none')
    axHisty.plot(y_bkgd_array,y_axis_array,color='r',ls='solid')
    axHisty.errorbar(y_bkgd_array,y_axis_array,xerr=y_syst_array,color='r',marker='.',ls='none')
    axHistx.errorbar(x_axis_array,x_count_array,yerr=x_error_array,color='k',marker='.',ls='none',label='Observation data')
    axHisty.errorbar(y_count_array,y_axis_array,xerr=y_error_array,color='k',marker='.',ls='none')
    axHistx.set_xlim(axHistx.get_xlim()[::-1])
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    axHistx.yaxis.set_major_formatter(formatter)
    axHistx.legend(loc='best')

    font = {'family': 'serif', 'color':  'white', 'weight': 'normal', 'size': 10, 'rotation': 0.,}
    lable_energy_range = f'E = {E_min:0.2f}-{E_max:0.2f} TeV'
    txt = axTemperature.text(xmax-0.14, ymax-0.21, lable_energy_range, fontdict=font)

    fig.savefig(f'output_plots/excess_{plotname}.png',bbox_inches='tight')



def PlotSkyMap(fig,label_z,logE_min,logE_max,hist_map_input,plotname,roi_x=[],roi_y=[],roi_r=[],excl_x=[],excl_y=[],excl_r=[],max_z=0.,colormap='coolwarm',layer=0,zoomin=1.0):

    E_min = pow(10.,logE_bins[logE_min])
    E_max = pow(10.,logE_bins[logE_max])

    hist_map = MakeSkymapCutout(hist_map_input,zoomin)

    xmin = hist_map.xaxis.min()
    xmax = hist_map.xaxis.max()
    ymin = hist_map.yaxis.min()
    ymax = hist_map.yaxis.max()

    other_stars, other_star_type, other_star_coord = GetGammaSourceInfo() 

    if coordinate_type == 'galactic':
        for star in range(0,len(other_stars)):
            other_star_coord[star][0], other_star_coord[star][1] = ConvertRaDecToGalactic(other_star_coord[star][0], other_star_coord[star][1])

    other_star_labels = []
    other_star_types = []
    other_star_markers = []
    star_range = 0.8*(xmax-xmin)/2.
    source_ra = (xmax+xmin)/2.
    source_dec = (ymax+ymin)/2.
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
    figsize_x = 6
    figsize_y = 7
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()

    label_x = 'RA [deg]'
    label_y = 'Dec [deg]'
    if coordinate_type == 'galactic':
        label_x = 'Gal. l [deg]'
        label_y = 'Gal. b [deg]'

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

    favorite_color = 'k'
    if colormap=='magma':
        favorite_color = 'deepskyblue'
    font = {'family': 'serif', 'color':  favorite_color, 'weight': 'normal', 'size': 10, 'rotation': 0.,}

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

    linestyles = ['-', '--', '-.', ':']  # List of linestyles
    for roi in range(0,len(roi_x)):
        mycircle = plt.Circle( (roi_x[roi], roi_y[roi]), roi_r[roi], fill = False, linestyle=linestyles[roi], color='white')
        axbig.add_patch(mycircle)
    for roi in range(0,len(excl_x)):
        mycircle = plt.Circle( (excl_x[roi], excl_y[roi]), excl_r[roi], fill = False, linestyle='-', color='black')
        axbig.add_patch(mycircle)

    if not 'Gas' in plotname:
        lable_energy_range = f'E = {E_min:0.2f}-{E_max:0.2f} TeV'
        txt = axbig.text(xmax-0.14, ymax-0.21, lable_energy_range, fontdict=font)

    fig.savefig(f'output_plots/{plotname}.png',bbox_inches='tight')
    axbig.remove()

def GetFluxCalibration(energy):

    if doFluxCalibration:
        return 1.

    flux_calibration = []
    for string in str_flux_calibration:
        flux_calibration.append(float(string))

    if flux_calibration[energy]>0.:
        return 1./flux_calibration[energy]
    else:
        return 0.

def make_significance_map(data_sky_map,bkgd_sky_map,significance_sky_map,excess_sky_map,syst_sky_map=None):
  
    skymap_bins = len(data_sky_map.xaxis)-1

    for idx_x in range(0,skymap_bins):
        for idx_y in range(0,skymap_bins):
            data = data_sky_map.waxis[idx_x,idx_y,0]
            bkgd = bkgd_sky_map.waxis[idx_x,idx_y,0]
            bkgd_err = 0.
            if syst_sky_map!=None:
                bkgd_err = syst_sky_map.waxis[idx_x,idx_y,0]
            significance = significance_li_and_ma(data, bkgd, bkgd_err)
            significance_sky_map.waxis[idx_x,idx_y,0] = significance
            excess_sky_map.waxis[idx_x,idx_y,0] = (data-bkgd)

def make_flux_map(incl_sky_map,data_sky_map,bkgd_sky_map,flux_sky_map,flux_err_sky_map,flux_syst_sky_map,avg_energy,delta_energy,syst_sky_map=None):
  
    skymap_bins = len(data_sky_map.xaxis)-1

    norm_content_max = np.max(incl_sky_map.waxis[:,:,0])
    norm_mean = np.mean(incl_sky_map.waxis[:,:,0])

    for idx_x in range(0,skymap_bins):
        for idx_y in range(0,skymap_bins):
            data = data_sky_map.waxis[idx_x,idx_y,0]
            norm = incl_sky_map.waxis[idx_x,idx_y,0]
            bkgd = bkgd_sky_map.waxis[idx_x,idx_y,0]
            bkgd_err_sq = 0.
            if syst_sky_map!=None:
                bkgd_err_sq = pow(syst_sky_map.waxis[idx_x,idx_y,0],2)
            if norm>0.:
                excess = data-bkgd
                error = pow(data,0.5)
                syst = pow(bkgd_err_sq,0.5)
                logE = logE_axis.get_bin(np.log10(avg_energy))
                correction = GetFluxCalibration(logE)/norm*pow(avg_energy,2)/(100.*100.*3600.)/delta_energy
                norm_weight = 1.
                flux = excess*correction*norm_weight
                flux_err = error*correction*norm_weight
                flux_syst = syst*correction*norm_weight
                flux_sky_map.waxis[idx_x,idx_y,0] = flux
                flux_err_sky_map.waxis[idx_x,idx_y,0] = flux_err
                flux_syst_sky_map.waxis[idx_x,idx_y,0] = flux_syst
            else:
                flux_sky_map.waxis[idx_x,idx_y,0] = 0.
                flux_err_sky_map.waxis[idx_x,idx_y,0] = 0.
                flux_syst_sky_map.waxis[idx_x,idx_y,0] = 0.

def GetRadialProfile(hist_flux_skymap,hist_error_skymap,hist_syst_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,use_excl=True,radial_bin_scale=0.1):

    deg2_to_sr =  3.046*1e-4
    pix_size = abs((hist_flux_skymap.yaxis[1]-hist_flux_skymap.yaxis[0])*(hist_flux_skymap.xaxis[1]-hist_flux_skymap.xaxis[0]))*deg2_to_sr
    bin_size = max(radial_bin_scale,1.*(hist_flux_skymap.yaxis[1]-hist_flux_skymap.yaxis[0]))
    radial_axis = MyArray1D(x_nbins=int(roi_r/bin_size),start_x=0.,end_x=roi_r)

    radius_array = []
    brightness_array = []
    brightness_err_array = []
    brightness_syst_array = []
    pixel_array = []
    for br in range(0,len(radial_axis.xaxis)-1):
        radius = 0.5*(radial_axis.xaxis[br]+radial_axis.xaxis[br+1])
        radius_array += [radius]
        brightness_array += [0.]
        brightness_err_array += [0.]
        brightness_syst_array += [0.]
        pixel_array += [0.]

    for br in range(0,len(radial_axis.xaxis)-1):
        radius = 0.5*(radial_axis.xaxis[br]+radial_axis.xaxis[br+1])
        for bx in range(0,len(hist_flux_skymap.xaxis)-1):
            for by in range(0,len(hist_flux_skymap.yaxis)-1):
                bin_ra = 0.5*(hist_flux_skymap.xaxis[bx]+hist_flux_skymap.xaxis[bx+1])
                bin_dec = 0.5*(hist_flux_skymap.yaxis[by]+hist_flux_skymap.yaxis[by+1])
                keep_event = False
                distance = pow(pow(bin_ra-roi_x[0],2) + pow(bin_dec-roi_y[0],2),0.5)
                if distance<radial_axis.xaxis[br+1] and distance>=radial_axis.xaxis[br]: 
                    keep_event = True
                if use_excl:
                    for roi in range(0,len(excl_roi_x)):
                        excl_distance = pow(pow(bin_ra-excl_roi_x[roi],2) + pow(bin_dec-excl_roi_y[roi],2),0.5)
                        #if excl_distance<excl_roi_r[roi]: 
                        #    keep_event = False
                if keep_event:
                    pixel_array[br] += 1.*pix_size
                    brightness_array[br] += hist_flux_skymap.waxis[bx,by,0]
                    if not hist_error_skymap==None:
                        brightness_err_array[br] += pow(hist_error_skymap.waxis[bx,by,0],2)
                    if not hist_syst_skymap==None:
                        brightness_syst_array[br] += hist_syst_skymap.waxis[bx,by,0]
        if pixel_array[br]==0.: continue
        brightness_array[br] = brightness_array[br]/pixel_array[br]
        brightness_err_array[br] = pow(brightness_err_array[br],0.5)/pixel_array[br]
        brightness_syst_array[br] = brightness_syst_array[br]/pixel_array[br]

    output_radius_array = []
    output_brightness_array = []
    output_brightness_err_array = []
    output_brightness_syst_array = []
    for br in range(0,len(radial_axis.xaxis)-1):
        radius = radius_array[br]
        brightness = brightness_array[br]
        brightness_err = brightness_err_array[br]
        brightness_syst = brightness_syst_array[br]
        output_radius_array += [radius]
        output_brightness_array += [brightness]
        output_brightness_err_array += [brightness_err]
        output_brightness_syst_array += [brightness_syst]

    return output_radius_array, output_brightness_array, output_brightness_err_array, output_brightness_syst_array

def GetRegionIntegral(hist_flux_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_error_skymap=None,hist_syst_skymap=None,use_excl=True):

    flux_sum = 0.
    flux_stat_err = 0.
    flux_syst_err = 0.
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
                flux_sum += hist_flux_skymap.waxis[bx,by,0]
                if hist_error_skymap==None:
                    flux_stat_err += 0.
                else:
                    flux_stat_err += pow(hist_error_skymap.waxis[bx,by,0],2)
                if hist_syst_skymap==None:
                    flux_syst_err += 0.
                else:
                    if np.isnan(hist_syst_skymap.waxis[bx,by,0]): continue
                    flux_syst_err += hist_syst_skymap.waxis[bx,by,0]
    flux_stat_err = pow(flux_stat_err,0.5)
    #flux_syst_err = pow(flux_syst_err,0.5)
    flux_err = pow(flux_stat_err*flux_stat_err + flux_syst_err*flux_syst_err,0.5)
    return flux_sum, flux_err

def GetRegionSpectrum(hist_flux_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_error_skymap=None,hist_syst_skymap=None,use_excl=True):

    x_axis = []
    x_error = []
    y_axis = []
    y_error = []

    binE_start = 0
    binE_end = logE_nbins

    for binE in range(binE_start,binE_end):
        flux_sum = 0.
        flux_stat_err = 0.
        if hist_error_skymap!=None and hist_syst_skymap!=None:
            flux_sum, flux_stat_err = GetRegionIntegral(hist_flux_skymap[binE],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_error_skymap=hist_error_skymap[binE],hist_syst_skymap=hist_syst_skymap[binE],use_excl=use_excl)
        elif hist_error_skymap!=None:
            flux_sum, flux_stat_err = GetRegionIntegral(hist_flux_skymap[binE],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_error_skymap=hist_error_skymap[binE],use_excl=use_excl)
        elif hist_syst_skymap!=None:
            flux_sum, flux_stat_err = GetRegionIntegral(hist_flux_skymap[binE],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_syst_skymap=hist_syst_skymap[binE],use_excl=use_excl)
        else:
            flux_sum, flux_stat_err = GetRegionIntegral(hist_flux_skymap[binE],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,use_excl=use_excl)
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

def GetHessGeminga():

    energies = [0.2044, 0.4838, 0.8421]
    fluxes = [-11.4839, -11.7228, -12.0616]
    flux_errs_up = [-11.4028, -11.6260, -11.9149]
    flux_errs_lo = [-11.5776, -11.8477, -12.2724]
    flux_errs = []

    erg_to_TeV = 0.62
    for entry in range(0,len(energies)):
        energies[entry] = pow(10.,energies[entry])
        fluxes[entry] = pow(10.,fluxes[entry])*erg_to_TeV
        flux_errs += [0.5*(pow(10.,flux_errs_up[entry])-pow(10.,flux_errs_lo[entry]))*erg_to_TeV]

    return energies, fluxes, flux_errs

def GetHawcDiffusionFluxJ1908():

    energies = [1.19,1.82,3.12,5.52,9.96,18.65,34.17,59.71,103.07,176.38]
    fluxes = [1.95e-11,1.98e-11,2.00e-11,1.57e-11,1.18e-11,7.19e-12,4.70e-12,2.75e-12,2.13e-12,1.38e-12]
    flux_errs = [1.95e-11,1.98e-11,2.00e-11,1.57e-11,1.18e-11,7.19e-12,4.70e-12,2.75e-12,2.13e-12,1.38e-12]
    flux_errs_up = [+0.14e-11,+0.14e-11,+0.13e-11,+0.09e-11,+0.07e-11,+0.55e-12,+0.46e-12,+0.43e-12,+0.44e-12,+0.54e-12]
    flux_errs_low = [-0.15e-11,-0.13e-11,-0.13e-11,-0.09e-11,-0.07e-11,-0.53e-11,-0.45e-12,-0.42e-12,-0.47e-12,-0.54e-12]

    for entry in range(0,len(energies)):
        energies[entry] = energies[entry]
        fluxes[entry] = fluxes[entry]/(energies[entry]*energies[entry])*pow(energies[entry],2)
        flux_errs_up[entry] = flux_errs[entry]+flux_errs_up[entry]
        flux_errs_low[entry] = flux_errs[entry]+flux_errs_low[entry]
        flux_errs[entry] = 0.25*fluxes[entry]

    return energies, fluxes, flux_errs

def flux_lhaaso_wcda_j1908_func(x):
    # TeV^{-1}cm^{-2}s^{-1}
    # https://arxiv.org/pdf/2305.17030.pdf
    Flux_N0 = 7.97 
    Gamma_index = 2.42
    return Flux_N0*pow(10,-13)*pow(x*1./3000.,-Gamma_index)

def GetHawcSaraFluxJ1908():

    #energies = [1.53,2.78,4.75,7.14,11.15,18.68,36.15,61.99,108.69,187.51]
    #fluxes = [7.0009e-12,9.5097e-12,8.4629e-12,6.6242e-12,5.6764e-12,4.4924e-12,3.2932e-12,1.5250e-12,9.1235e-13,4.1833e-13]
    #flux_errs = [7.0009e-12,9.5097e-12,8.4629e-12,6.6242e-12,5.6764e-12,4.4924e-12,3.2932e-12,1.5250e-12,9.1235e-13,4.1833e-13]
    #flux_errs_up = [+7.2024e-13,+6.3288e-13,+5.4679e-13,+3.9318e-13,+2.6768e-13,+2.9978e-13,+2.2130e-13,+1.8650e-13,+1.8756e-13,+1.5458e-13]
    #flux_errs_low = [-7.1498e-13,-6.6198e-13,-5.2961e-13,-3.8152e-13,-2.8404e-13,-3.1157e-13,-2.0721e-13,-1.8818e-13,-1.7827e-13,-1.5612e-13]

    energies = [1.53,2.78,4.75,7.14,11.15,18.68,36.15,61.99,108.69,187.51]
    fluxes = [7.0009e-12,9.5097e-12,8.4629e-12,6.6242e-12,5.6764e-12,4.4924e-12,3.2932e-12,1.5250e-12,9.1235e-13,4.1833e-13]
    flux_errs = [7.0009e-12,9.5097e-12,8.4629e-12,6.6242e-12,5.6764e-12,4.4924e-12,3.2932e-12,1.5250e-12,9.1235e-13,4.1833e-13]
    flux_errs_low = [-1.293e-12,-1.425e-12,-1.623e-12,-1.292e-12,-8.837e-13,-4.499e-13,-4.926e-13,-2.418e-13,-1.947e-13,-1.549e-13]
    flux_errs_up = [+8.199e-13,+6.952e-13,+7.313e-13,+5.473e-13,+4.411e-13,+4.389e-13,+3.015e-13,+1.893e-13,+2.074e-13,+1.649e-13]

    for entry in range(0,len(energies)):
        energies[entry] = energies[entry]
        fluxes[entry] = fluxes[entry]/(energies[entry]*energies[entry])*pow(energies[entry],2)
        flux_errs_up[entry] = flux_errs[entry]+flux_errs_up[entry]
        flux_errs_low[entry] = flux_errs[entry]+flux_errs_low[entry]
        flux_errs[entry] = 0.5*(flux_errs_up[entry]-flux_errs_low[entry])/(energies[entry]*energies[entry])*pow(energies[entry],2)

    return energies, fluxes, flux_errs

def GetHessFluxJ1908():
    energies = [pow(10.,-0.332),pow(10.,0.022),pow(10.,0.396),pow(10.,0.769),pow(10.,1.124),pow(10.,1.478)]
    fluxes = [pow(10.,-10.981),pow(10.,-10.967),pow(10.,-11.057),pow(10.,-11.169),pow(10.,-11.188),pow(10.,-11.386)]
    flux_errs = [pow(10.,-0.332),pow(10.,0.022),pow(10.,0.396),pow(10.,0.769),pow(10.,1.124),pow(10.,1.478)]
    flux_errs_up = [pow(10.,-10.895),pow(10.,-10.916),pow(10.,-11.003),pow(10.,-11.101),pow(10.,-11.101),pow(10.,-11.264)]
    flux_errs_low = [pow(10.,-11.086),pow(10.,-11.010),pow(10.,-11.126),pow(10.,-11.264),pow(10.,-11.292),pow(10.,-11.556)]

    for entry in range(0,len(energies)):
        energies[entry] = energies[entry]
        fluxes[entry] = fluxes[entry]/(energies[entry]*energies[entry])*pow(energies[entry],2)
        flux_errs[entry] = 0.5*(flux_errs_up[entry]-flux_errs_low[entry])/(energies[entry]*energies[entry])*pow(energies[entry],2)

    return energies, fluxes, flux_errs

def GetFermiJordanFluxJ1908():

    energies = [42571.11253606245,85723.52082084052,172617.57055787765,347592.1821687443]
    fluxes = [2.856783157929038e-06,3.89109583469775e-06,5.0680678657082445e-06,9.271213817855382e-06]
    flux_stat_errs = [1.1604625384485099e-06,1.556189798998829e-06,2.2448723890895238e-06,3.4737117958614837e-06]
    flux_syst_errs = [1.135182978267407e-06,7.805371450450492e-07,1.6102184866176414e-06,1.5283362877401339e-06]
    flux_errs = []

    for entry in range(0,len(energies)):
        energies[entry] = energies[entry]/1e6
        fluxes[entry] = fluxes[entry]/1e6
        flux_stat_errs[entry] = flux_stat_errs[entry]/1e6
        flux_syst_errs[entry] = flux_syst_errs[entry]/1e6
        flux_errs += [pow(pow(flux_stat_errs[entry],2)+pow(flux_syst_errs[entry],2),0.5)]

    return energies, fluxes, flux_errs

def GetLHAASOFluxJ1908():
    energies = [pow(10.,1.102),pow(10.,1.302),pow(10.,1.498),pow(10.,1.700),pow(10.,1.900),pow(10.,2.099),pow(10.,2.299),pow(10.,2.498),pow(10.,2.697)]
    fluxes = [pow(10.,-11.033),pow(10.,-10.988),pow(10.,-11.201),pow(10.,-11.324),pow(10.,-11.553),pow(10.,-11.860),pow(10.,-11.921),pow(10.,-12.346),pow(10.,-12.653)]
    flux_errs = [pow(10.,1.102),pow(10.,1.302),pow(10.,1.498),pow(10.,1.700),pow(10.,1.900),pow(10.,2.099),pow(10.,2.299),pow(10.,2.498),pow(10.,2.697)]
    flux_errs_up = [pow(10.,-10.966),pow(10.,-10.949),pow(10.,-11.167),pow(10.,-11.296),pow(10.,-11.513),pow(10.,-11.798),pow(10.,-11.854),pow(10.,-12.173),pow(10.,-12.391)]
    flux_errs_low = [pow(10.,-11.094),pow(10.,-11.027),pow(10.,-11.240),pow(10.,-11.368),pow(10.,-11.597),pow(10.,-11.944),pow(10.,-12.022),pow(10.,-12.536),pow(10.,-13.128)]

    erg_to_TeV = 0.62
    for entry in range(0,len(energies)):
        fluxes[entry] = fluxes[entry]*erg_to_TeV/(energies[entry]*energies[entry])*pow(energies[entry],2)
        flux_errs[entry] = 0.5*(flux_errs_up[entry]-flux_errs_low[entry])*erg_to_TeV/(energies[entry]*energies[entry])*pow(energies[entry],2)

    return energies, fluxes, flux_errs

def GetHessJ1857():

    energies = [400.0/1000., 950.0/1000., 2260.0/1000., 5360.0/1000., 12710.0/1000.]
    fluxes = [1.53e-11, 1.13e-11, 6.46e-12, 3.87e-12, 8.58e-13]
    flux_errs_lo = [1.67e-12, 9.37e-13, 9.07e-13, 9.46e-13, 8.52e-13]
    flux_errs_up = [1.72e-12, 9.21e-13, 9.15e-13, 9.79e-13, 9.27e-13]
    flux_errs = []

    erg_to_TeV = 0.62
    for entry in range(0,len(energies)):
        fluxes[entry] = fluxes[entry]*erg_to_TeV
        flux_errs += [0.5*(flux_errs_up[entry]+flux_errs_lo[entry])*erg_to_TeV]

    return energies, fluxes, flux_errs

def GetMagicJ1857():

    energies = [100.597/1000., 172.933/1000., 297.285/1000., 511.054/1000., 878.539/1000., 1510.27/1000., 2596.27/1000., 4463.17/1000., 7672.51/1000., 13189.6/1000.]
    fluxes = [1.25e-11, 8.17e-12, 1.09e-11, 9.22e-12, 9.46e-12, 9.12e-12, 7.57e-12, 4.21e-12, 3.79e-12, 7.77e-12]
    flux_errs_lo = [7.96e-12, 2.51e-12, 2.58e-12, 2.14e-12, 2.08e-12, 2.12e-12, 2.63e-12, 2.14e-12, 1.78e-12, 4.63e-12]
    flux_errs_up = [7.96e-12, 2.51e-12, 2.58e-12, 2.14e-12, 2.08e-12, 2.12e-12, 2.63e-12, 2.14e-12, 1.78e-12, 4.63e-12]
    flux_errs = []

    erg_to_TeV = 0.62
    for entry in range(0,len(energies)):
        fluxes[entry] = fluxes[entry]*erg_to_TeV
        flux_errs += [0.5*(flux_errs_up[entry]+flux_errs_lo[entry])*erg_to_TeV]

    return energies, fluxes, flux_errs

def GetHAWCDiffusionFluxGeminga():

    energies = [pow(10.,0.90),pow(10.,1.60)]
    fluxes = [pow(10.,-11.12),pow(10.,-11.36)]
    flux_errs = [0.,0.]
    flux_errs_up = [pow(10.,-11.04),pow(10.,-11.28)]
    flux_errs_low = [pow(10.,-11.21),pow(10.,-11.44)]

    for entry in range(0,len(energies)):
        fluxes[entry] = fluxes[entry]/(energies[entry]*energies[entry])*pow(energies[entry],2)
        flux_errs[entry] = 0.5*(flux_errs_up[entry]-flux_errs_low[entry])/(energies[entry]*energies[entry])*pow(energies[entry],2)

    return energies, fluxes, flux_errs

def GetHAWCGaussianFluxGeminga():

    energies = [pow(10.,0.90),pow(10.,1.60)]
    fluxes = [pow(10.,-11.36),pow(10.,-11.52)]
    flux_errs = [0.,0.]
    flux_errs_up = [pow(10.,-11.28),pow(10.,-11.45)]
    flux_errs_low = [pow(10.,-11.44),pow(10.,-11.59)]

    for entry in range(0,len(energies)):
        fluxes[entry] = fluxes[entry]/(energies[entry]*energies[entry])*pow(energies[entry],2)
        flux_errs[entry] = 0.5*(flux_errs_up[entry]-flux_errs_low[entry])/(energies[entry]*energies[entry])*pow(energies[entry],2)

    return energies, fluxes, flux_errs

def GetHAWCDiskFluxGeminga():

    energies = [pow(10.,0.00),pow(10.,1.70)]
    fluxes = [pow(10.,-11.42),pow(10.,-11.81)]
    flux_errs = [0.,0.]
    flux_errs_up = [pow(10.,-11.30),pow(10.,-11.68)]
    flux_errs_low = [pow(10.,-11.56),pow(10.,-11.94)]

    for entry in range(0,len(energies)):
        fluxes[entry] = fluxes[entry]/(energies[entry]*energies[entry])*pow(energies[entry],2)
        flux_errs[entry] = 0.5*(flux_errs_up[entry]-flux_errs_low[entry])/(energies[entry]*energies[entry])*pow(energies[entry],2)

    return energies, fluxes, flux_errs

def GetFermiFluxGeminga():

    energies = [pow(10.,1.03),pow(10.,1.30),pow(10.,1.56),pow(10.,1.82)]
    fluxes = [pow(10.,-7.27),pow(10.,-7.29),pow(10.,-7.41),pow(10.,-7.35)]
    flux_errs = [pow(10.,-7.27),pow(10.,-7.29),pow(10.,-7.41),pow(10.,-7.35)]
    flux_errs_up = [pow(10.,-7.14),pow(10.,-7.16),pow(10.,-7.29),pow(10.,-7.23)]
    flux_errs_low = [pow(10.,-7.46),pow(10.,-7.49),pow(10.,-7.58),pow(10.,-7.50)]

    GeV_to_TeV = 1e-3
    for entry in range(0,len(energies)):
        fluxes[entry] = fluxes[entry]*GeV_to_TeV/(energies[entry]*energies[entry]/1e6)*pow(energies[entry]/1e3,2)
        flux_errs[entry] = 0.5*(flux_errs_up[entry]-flux_errs_low[entry])*GeV_to_TeV/(energies[entry]*energies[entry]/1e6)*pow(energies[entry]/1e3,2)
        energies[entry] = energies[entry]/1000.

    return energies, fluxes, flux_errs

def GetFermiUpperLimitFluxGeminga():

    energies = [pow(10.,2.09),pow(10.,2.35),pow(10.,2.61),pow(10.,2.87)]
    fluxes = [pow(10.,-7.35),pow(10.,-7.23),pow(10.,-7.34),pow(10.,-7.18)]
    fluxes_err = [pow(10.,-7.35),pow(10.,-7.23),pow(10.,-7.34),pow(10.,-7.18)]

    GeV_to_TeV = 1e-3
    for entry in range(0,len(energies)):
        fluxes[entry] = fluxes[entry]*GeV_to_TeV/(energies[entry]*energies[entry]/1e6)*pow(energies[entry]/1e3,2)
        fluxes_err[entry] = fluxes[entry]*0.3
        energies[entry] = energies[entry]/1000.

    return energies, fluxes, fluxes_err


def PrintAndPlotInformationRoI(fig,logE_min,logE_mid,logE_max,source_name,hist_data_skymap,hist_bkgd_skymap,hist_syst_skymap,hist_flux_skymap,hist_flux_err_skymap,hist_flux_syst_skymap,hist_mimic_data_skymap,hist_mimic_bkgd_skymap,roi_name,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r):

    energy_axis, energy_error, flux, flux_incl_err = GetRegionSpectrum(hist_flux_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_error_skymap=hist_flux_err_skymap,hist_syst_skymap=hist_flux_syst_skymap)
    energy_axis, energy_error, data, data_stat_err = GetRegionSpectrum(hist_data_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)
    energy_axis, energy_error, bkgd, bkgd_syst_err = GetRegionSpectrum(hist_bkgd_skymap,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,hist_syst_skymap=hist_syst_skymap)

    bkgd_incl_err = np.zeros_like(bkgd_syst_err)
    n_mimic = len(hist_mimic_data_skymap)
    list_mimic_data = []
    list_mimic_bkgd = []
    for mimic in range(0,n_mimic):
        mimic_energy_axis, mimic_energy_error, mimic_data, mimic_data_stat_err = GetRegionSpectrum(hist_mimic_data_skymap[mimic],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)
        mimic_energy_axis, mimic_energy_error, mimic_bkgd, mimic_bkgd_stat_err = GetRegionSpectrum(hist_mimic_bkgd_skymap[mimic],roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)
        list_mimic_data += [mimic_data]
        list_mimic_bkgd += [mimic_bkgd]
    for binx in range(0,len(energy_axis)):
        on_data = data[binx]
        stat_err = pow(on_data,0.5)
        syst_err = bkgd_syst_err[binx]
        #syst_err = 0.
        #for mimic in range(0,n_mimic):
        #    if list_mimic_data[mimic][binx]==0.: continue
        #    # syst error is measured as a relative error
        #    mimic_data = list_mimic_data[mimic][binx]
        #    mimic_bkgd = list_mimic_bkgd[mimic][binx]
        #    mimic_stat_err = pow(mimic_data,0.5)
        #    syst_err += max(0.,pow(mimic_data-mimic_bkgd,2)-pow(mimic_stat_err,2))/pow(mimic_data,2)
        #if n_mimic>0:
        #    syst_err = syst_err*pow(on_data,2)
        #    syst_err = pow(syst_err/float(n_mimic),0.5)
        #bkgd_syst_err[binx] = syst_err
        data_stat_err[binx] = stat_err
        bkgd_incl_err[binx] = syst_err


    vectorize_f_crab = np.vectorize(flux_crab_func)
    ydata_crab_ref = pow(np.array(energy_axis),2)*vectorize_f_crab(energy_axis)

    flux_floor = []
    flux_cu = []
    flux_err_cu = []
    for binx in range(0,len(energy_axis)):
        if flux[binx]>0.:
            flux_floor += [flux[binx]]
            flux_cu += [flux[binx]/ydata_crab_ref[binx]]
            flux_err_cu += [flux_incl_err[binx]/ydata_crab_ref[binx]]
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
    print (f'RoI info: {roi_name[0]}, roi_x = {roi_x[0]}, roi_y = {roi_y[0]}, roi_r = {roi_r[0]}')

    min_energy = pow(10.,logE_bins[logE_min])
    mid_energy = pow(10.,logE_bins[logE_mid])
    max_energy = pow(10.,logE_bins[logE_max])

    sum_data = 0.
    sum_bkgd = 0.
    sum_error = 0.
    avg_energy = 0.
    sum_flux = 0.
    for binx in range(0,len(energy_axis)):
        if energy_axis[binx]>min_energy and energy_axis[binx]<max_energy:
            sum_data += data[binx]
            sum_bkgd += bkgd[binx]
            sum_error += bkgd_incl_err[binx]*bkgd_incl_err[binx]
            avg_energy += flux[binx]*energy_axis[binx]
            sum_flux += flux[binx]
    sum_error = pow(sum_error,0.5)
    significance = significance_li_and_ma(sum_data,sum_bkgd,sum_error)
    avg_energy = avg_energy/sum_flux
    print (f'E = {min_energy:0.2f}-{max_energy:0.2f} TeV, avg_E = {avg_energy:0.2f} TeV, data = {sum_data:0.1f}, bkgd = {sum_bkgd:0.1f} +/- {sum_error:0.1f}, significance = {significance:0.1f} sigma')

    sum_data = 0.
    sum_bkgd = 0.
    sum_error = 0.
    avg_energy = 0.
    sum_flux = 0.
    for binx in range(0,len(energy_axis)):
        if energy_axis[binx]>min_energy and energy_axis[binx]<mid_energy:
            sum_data += data[binx]
            sum_bkgd += bkgd[binx]
            sum_error += bkgd_incl_err[binx]*bkgd_incl_err[binx]
            avg_energy += flux[binx]*energy_axis[binx]
            sum_flux += flux[binx]
    sum_error = pow(sum_error,0.5)
    significance = significance_li_and_ma(sum_data,sum_bkgd,sum_error)
    avg_energy = avg_energy/sum_flux
    print (f'E = {min_energy:0.2f}-{mid_energy:0.2f} TeV, avg_E = {avg_energy:0.2f} TeV, data = {sum_data:0.1f}, bkgd = {sum_bkgd:0.1f} +/- {sum_error:0.1f}, significance = {significance:0.1f} sigma')

    sum_data = 0.
    sum_bkgd = 0.
    sum_error = 0.
    avg_energy = 0.
    sum_flux = 0.
    for binx in range(0,len(energy_axis)):
        if energy_axis[binx]>mid_energy and energy_axis[binx]<max_energy:
            sum_data += data[binx]
            sum_bkgd += bkgd[binx]
            sum_error += bkgd_incl_err[binx]*bkgd_incl_err[binx]
            avg_energy += flux[binx]*energy_axis[binx]
            sum_flux += flux[binx]
    sum_error = pow(sum_error,0.5)
    significance = significance_li_and_ma(sum_data,sum_bkgd,sum_error)
    avg_energy = avg_energy/sum_flux
    print (f'E = {mid_energy:0.2f}-{max_energy:0.2f} TeV, avg_E = {avg_energy:0.2f} TeV, data = {sum_data:0.1f}, bkgd = {sum_bkgd:0.1f} +/- {sum_error:0.1f}, significance = {significance:0.1f} sigma')

    for binx in range(0,len(energy_axis)):
        significance = significance_li_and_ma(data[binx],bkgd[binx],bkgd_incl_err[binx])
        print (f'E = {energy_axis[binx]:0.2f} ({pow(10.,logE_bins[binx]):0.2f}-{pow(10.,logE_bins[binx+1]):0.2f})TeV, data = {data[binx]:0.1f} +/- {data_stat_err[binx]:0.1f}, bkgd = {bkgd[binx]:0.1f} +/- {bkgd_incl_err[binx]:0.1f}, flux = {flux[binx]:0.2e} +/- {flux_incl_err[binx]:0.2e} TeV/cm2/s ({flux_cu[binx]:0.2f} CU), significance = {significance:0.1f} sigma')
    print ('===============================================================================================================')

    cu_uplims = np.zeros_like(energy_axis)
    for x in range(0,len(flux)):
        if flux_err_cu[x]==0.: continue
        significance = flux_cu[x]/flux_err_cu[x]
        if significance<2.:
            cu_uplims[x] = 1.
            flux_cu[x] = max(2.*flux_err_cu[x],flux_cu[x]+2.*flux_err_cu[x])

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
    axbig.errorbar(energy_axis,flux_cu,flux_err_cu,xerr=energy_error,uplims=cu_uplims,color='k',marker='_',ls='none',zorder=1)
    fig.savefig(f'output_plots/{source_name}_roi_flux_crab_unit_{roi_name[0]}.png',bbox_inches='tight')
    axbig.remove()

    PrintSpectralDataForNaima(energy_axis,flux,flux_incl_err,f'VERITAS_{roi_name[0]}')

    uplims = np.zeros_like(energy_axis)
    for x in range(0,len(flux)):
        if flux_incl_err[x]==0.: continue
        significance = flux[x]/flux_incl_err[x]
        if significance<2.:
            uplims[x] = 1.
            flux[x] = max(2.*flux_incl_err[x],flux[x]+2.*flux_incl_err[x])
            #flux_incl_err[x] = flux[x]

    fig.clf()
    figsize_x = 7
    figsize_y = 5
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'Energy [TeV]'
    label_y = '$E^{2}$ dN/dE [$\\mathrm{TeV}\\cdot\\mathrm{cm}^{-2}\\mathrm{s}^{-1}$]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.set_xscale('log')
    axbig.set_yscale('log')
    #axbig.fill_between(energy_axis,np.array(flux_floor)-np.array(flux_incl_err),np.array(flux_floor)+np.array(flux_incl_err),alpha=0.2,color='b',zorder=0)
    axbig.errorbar(energy_axis,flux,flux_incl_err,xerr=energy_error,uplims=uplims,color='k',marker='_',ls='none',label=f'VERITAS ({roi_name[1]})',zorder=1)
    if 'SS433' in source_name:
        HessSS433e_energies, HessSS433e_fluxes, HessSS433e_flux_errs = GetHessSS433e()
        HessSS433w_energies, HessSS433w_fluxes, HessSS433w_flux_errs = GetHessSS433w()
        axbig.errorbar(HessSS433e_energies,HessSS433e_fluxes,HessSS433e_flux_errs,marker='s',ls='none',label='HESS eastern',zorder=2)
        axbig.errorbar(HessSS433w_energies,HessSS433w_fluxes,HessSS433w_flux_errs,marker='s',ls='none',label='HESS western',zorder=3)
    elif 'PSR_J1856_p0245' in source_name:
        HESS_energies, HESS_fluxes, HESS_flux_errs = GetHessJ1857()
        axbig.errorbar(HESS_energies,HESS_fluxes,HESS_flux_errs,color='r',marker='_',ls='none',label='HESS')
        Magic_energies, Magic_fluxes, Magic_flux_errs = GetMagicJ1857()
        axbig.errorbar(Magic_energies,Magic_fluxes,Magic_flux_errs,color='g',marker='_',ls='none',label='MAGIC')
    elif 'PSR_J1907_p0602' in source_name:
        HAWC_energies, HAWC_fluxes, HAWC_flux_errs = GetHawcDiffusionFluxJ1908()
        Sara_energies, Sara_fluxes, Sara_flux_errs = GetHawcSaraFluxJ1908()
        HESS_energies, HESS_fluxes, HESS_flux_errs = GetHessFluxJ1908()
        Jordan_energies, Jordan_fluxes, Jordan_flux_errs = GetFermiJordanFluxJ1908()
        LHAASO_energies, LHAASO_fluxes, LHAASO_flux_errs = GetLHAASOFluxJ1908()
        axbig.errorbar(Jordan_energies,Jordan_fluxes,Jordan_flux_errs,color='g',marker='s',ls='none',label='Fermi-LAT',zorder=1)
        axbig.errorbar(Sara_energies,Sara_fluxes,Sara_flux_errs,color='purple',marker='o',ls='none',label='HAWC',zorder=5)
        axbig.errorbar(LHAASO_energies,LHAASO_fluxes,LHAASO_flux_errs,color='goldenrod',marker='^',ls='none',label='LHAASO (KM2A)',zorder=9)
    elif 'Geminga' in source_name:
        HAWC_diff_energies, HAWC_diff_fluxes, HAWC_diff_flux_errs = GetHAWCDiffusionFluxGeminga()
        HAWC_disk_energies, HAWC_disk_fluxes, HAWC_disk_flux_errs = GetHAWCDiskFluxGeminga()
        HAWC_gaus_energies, HAWC_gaus_fluxes, HAWC_gaus_flux_errs = GetHAWCGaussianFluxGeminga()
        HESS_energies, HESS_fluxes, HESS_flux_errs = GetHessGeminga()
        Fermi_energies, Fermi_fluxes, Fermi_flux_errs = GetFermiFluxGeminga()
        Fermi_UL_energies, Fermi_UL_fluxes, Fermi_UL_err = GetFermiUpperLimitFluxGeminga()
        axbig.plot(HAWC_diff_energies, HAWC_diff_fluxes,'g-',label='HAWC diffusion')
        axbig.fill_between(HAWC_diff_energies, np.array(HAWC_diff_fluxes)-np.array(HAWC_diff_flux_errs), np.array(HAWC_diff_fluxes)+np.array(HAWC_diff_flux_errs), alpha=0.2, color='g')
        axbig.plot(HAWC_disk_energies, HAWC_disk_fluxes,'m-',label='HAWC disk')
        axbig.fill_between(HAWC_disk_energies, np.array(HAWC_disk_fluxes)-np.array(HAWC_disk_flux_errs), np.array(HAWC_disk_fluxes)+np.array(HAWC_disk_flux_errs), alpha=0.2, color='m')
        axbig.plot(HAWC_gaus_energies, HAWC_gaus_fluxes,'y-',label='HAWC gaussian')
        axbig.fill_between(HAWC_gaus_energies, np.array(HAWC_gaus_fluxes)-np.array(HAWC_gaus_flux_errs), np.array(HAWC_gaus_fluxes)+np.array(HAWC_gaus_flux_errs), alpha=0.2, color='y')
        axbig.errorbar(HESS_energies,HESS_fluxes,HESS_flux_errs,color='b',marker='_',ls='none',label='HESS (1.0 deg)')
        axbig.errorbar(Fermi_energies,Fermi_fluxes,Fermi_flux_errs,color='r',marker='_',ls='none',label='Fermi')
        fermi_uplims = np.array([1,1,1,1], dtype=bool)
        axbig.errorbar(Fermi_UL_energies,Fermi_UL_fluxes,Fermi_UL_err,color='r',marker='_',ls='none',uplims=fermi_uplims)
    axbig.legend(loc='best')
    fig.savefig(f'output_plots/{source_name}_roi_energy_flux_{roi_name[0]}.png',bbox_inches='tight')
    axbig.remove()

def DefineRegionOfMask(src_name,src_ra,src_dec):

    region_x = [src_ra]
    region_y = [src_dec]
    region_r = [fov_mask_radius]
    region_name = ['center']

    if 'Crab' in src_name:
        region_x = [src_ra]
        region_y = [src_dec]
        region_r = [0.3]
        region_name = ['center']

    else:

        gamma_source_coord = GetGammaSources(src_ra,src_dec)
        for src in range(0,len(gamma_source_coord)):
            src_x = gamma_source_coord[src][0]
            src_y = gamma_source_coord[src][1]
            region_x += [src_x]
            region_y += [src_y]
            region_r += [0.3]
            region_name += ['TeVCat']

    return region_name, region_x, region_y, region_r

def DefineRegionOfExclusion(src_name,src_ra,src_dec,coordinate_type='icrs'):

    region_x = [src_ra]
    region_y = [src_dec]
    region_r = [0.]
    region_name = ['center']


    if coordinate_type == 'galactic':
        for roi in range(0,len(region_r)):
            src_x = region_x[roi]
            src_y = region_y[roi]
            src_x, src_y = ConvertRaDecToGalactic(src_x, src_y)
            region_x[roi] = src_x
            region_y[roi] = src_y
    elif coordinate_type == 'relative':
        for roi in range(0,len(region_r)):
            src_x = region_x[roi]
            src_y = region_y[roi]
            region_x[roi] = src_x - src_ra
            region_y[roi] = src_y - src_dec

    return region_name, region_x, region_y, region_r



def DefineRegionOfInterest(src_name,src_ra,src_dec,coordinate_type='icrs'):

    region_name = ('default','default region')
    region_x = []
    region_y = []
    region_r = []

    src_x = src_ra
    src_y = src_dec

    if coordinate_type == 'relative':

        region_name = ('center','center')
        region_x += [0.]
        region_y += [0.]
        region_r += [3.]

    elif 'Crab' in src_name:

        region_name = ('center','center')
        region_x += [src_x]
        region_y += [src_y]
        region_r += [calibration_radius]

        #region_name = ('star','star')
        #src_x = 84.4
        #src_y = 21.1
        #region_x += [src_x]
        #region_y += [src_y]
        #region_r += [calibration_radius]

    elif 'Geminga' in src_name:

        region_name = ('1p5deg','1.5 deg')
        region_x += [src_x]
        region_y += [src_y]
        region_r += [1.5]

        #region_name = ('1p0deg','1.0 deg')
        #region_x += [src_x]
        #region_y += [src_y]
        #region_r += [1.0]

    elif 'SNR_G189_p03' in src_name:

        region_name = ('SNR','SNR')
        src_x = 94.25
        src_y = 22.57
        region_x += [src_x]
        region_y += [src_y]
        region_r += [0.5]

        #region_name = ('HAWC','HAWC')
        #src_x = 94.25
        #src_y = 22.57
        #region_x += [src_x]
        #region_y += [src_y]
        #region_r += [1.05]

    elif 'PSR_J2021_p4026' in src_name:

        #region_name = ('SNR_full','SNR (full)')
        #src_x = 305.21
        #src_y = 40.43
        #region_x += [src_x]
        #region_y += [src_y]
        #region_r += [1.0]

        region_name = ('SNR_core','SNR (core)')
        src_x = 305.21
        src_y = 40.43
        region_x += [src_x]
        region_y += [src_y]
        region_r += [0.5]

        #region_name = ('SNR_shell','SNR (shell)')
        #src_x = 305.21
        #src_y = 40.43
        #region_x += [src_x]
        #region_y += [src_y]
        #region_r += [1.0]

    elif 'PSR_J1907_p0602' in src_name:

        #region_name = ('3HWC','3HWC')
        #region_x += [287.05]
        #region_y += [6.39]
        #region_r += [1.2]

        #region_name = ('SS433','SS 433')
        #region_x += [288.0833333]
        #region_y += [4.9166667]
        #region_r += [0.2]

        region_name = ('J1907_p0602_full','J1907+0602 (full)')
        src_x = 286.98
        src_y = 6.04
        region_x += [src_x]
        region_y += [src_y]
        region_r += [1.2]

        #region_name = ('J1907_p0602_inner','J1907+0602 (inner)')
        #src_x = 286.98
        #src_y = 6.04
        #region_x += [src_x]
        #region_y += [src_y]
        #region_r += [0.46]


    elif 'PSR_J1856_p0245' in src_name:

        #region_name = ('J1856_p0245_full','J1856+0245 (full)')
        #src_x = 284.21
        #src_y = 2.76
        #region_x += [src_x]
        #region_y += [src_y]
        #region_r += [1.0]

        #region_name = ('J1856_p0245_inner','J1856+0245 (inner)')
        #src_x = 284.21
        #src_y = 2.76
        #region_x += [src_x]
        #region_y += [src_y]
        #region_r += [0.15]

        #region_name = ('J1856_p0245_outer','J1856+0245 (outer)')
        #src_x = 284.21
        #src_y = 2.76
        #region_x += [src_x]
        #region_y += [src_y]
        #region_r += [1.0]

        region_name = ('0p4_deg','0.4 deg') # MAGIC ROI
        src_x = 284.3
        src_y = 2.7
        region_x += [src_x]
        region_y += [src_y]
        region_r += [0.4]

        #region_name = ('J1858_p020','J1858+020')
        #region_x += [284.6]
        #region_y += [2.1]
        #region_r += [0.2]

        #region_name = ('W44','W44')
        #region_x += [284.00]
        #region_y += [1.37]
        #region_r += [0.25]

    elif 'SS433' in src_name:
    
        region_name = ('3HWC','3HWC')
        region_x += [287.05]
        region_y += [6.39]
        region_r += [1.2]

        #region_name = ('SNR','SS 433 SNR')
        #region_x += [288.0833333]
        #region_y += [4.9166667]
        #region_r += [0.2]
    
        #region_name = ('west','SS 433 west')
        #region_x += [287.45138775]
        #region_y += [5.06731983]
        #region_r += [0.2]
    
        #region_name = ('east','SS 433 east')
        #region_x += [288.38690451]
        #region_y += [5.00610516]
        #region_r += [0.2]

    elif 'PSR_J2030_p4415' in src_name:
    
        region_name = ('1p5deg','1.5-deg diameter')
        region_x += [src_x]
        region_y += [src_y]
        region_r += [1.5]

    else:

        region_name = ('default','default')
        region_x += [src_x]
        region_y += [src_y]
        region_r += [3.0]


    if coordinate_type == 'galactic':
        for roi in range(0,len(region_r)):
            region_x[roi], region_y[roi] = ConvertRaDecToGalactic(region_x[roi], region_y[roi])

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

def diffusion_func(x,A,d):
    return A*1.22/(pow(3.14,1.5)*d*(x+0.06*d))*np.exp(-x*x/(d*d))

def plot_radial_profile_with_systematics(fig,plotname,logE_min,logE_max,flux_sky_map,flux_err_sky_map,mimic_flux_sky_map,mimic_flux_err_sky_map,roi_x,roi_y,excl_roi_x,excl_roi_y,excl_roi_r,fit_radial_profile,radial_bin_scale=0.1):

    E_min = pow(10.,logE_bins[logE_min])
    E_max = pow(10.,logE_bins[logE_max])

    on_radial_axis, on_profile_axis, on_profile_err_axis = GetRadialProfile(flux_sky_map,flux_err_sky_map,roi_x,roi_y,2.0,excl_roi_x,excl_roi_y,excl_roi_r,radial_bin_scale=radial_bin_scale)
    all_radial_axis, all_profile_axis, all_profile_err_axis = GetRadialProfile(flux_sky_map,flux_err_sky_map,roi_x,roi_y,2.0,excl_roi_x,excl_roi_y,excl_roi_r,use_excl=False,radial_bin_scale=radial_bin_scale)

    n_mimic = len(mimic_flux_sky_map)
    fig.clf()
    figsize_x = 7
    figsize_y = 5
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'angular distance [deg]'
    label_y = 'surface brightness [$\\mathrm{TeV}\\ \\mathrm{cm}^{-2}\\mathrm{s}^{-1}\\mathrm{sr}^{-1}$]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    mimic_profile_axis = []
    mimic_profile_err_axis = []
    for mimic in range(0,n_mimic):
        radial_axis, profile_axis, profile_err_axis = GetRadialProfile(mimic_flux_sky_map[mimic],mimic_flux_err_sky_map[mimic],roi_x,roi_y,2.0,excl_roi_x,excl_roi_y,excl_roi_r,use_excl=False,radial_bin_scale=radial_bin_scale)
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
        if n_mimic>0:
            syst_err = pow(syst_err/float(n_mimic),0.5)
        profile_syst_err_axis += [syst_err]
    baseline_yaxis = [0. for i in range(0,len(on_radial_axis))]
    axbig.plot(on_radial_axis, baseline_yaxis, color='b', ls='dashed')
    axbig.fill_between(on_radial_axis,-np.array(profile_syst_err_axis),np.array(profile_syst_err_axis),alpha=0.2,color='b',zorder=0)
    fig.savefig(f'output_plots/{plotname}_mimic.png',bbox_inches='tight')
    axbig.remove()

    if fit_radial_profile:

        curve_fit_radius_array = []
        curve_fit_brightness_array = []
        curve_fit_brightness_err_array = []
        for br in range(0,len(on_radial_axis)):
            radius = on_radial_axis[br]
            brightness = on_profile_axis[br]
            brightness_err = on_profile_err_axis[br]
            if radius>1.7: continue
            curve_fit_radius_array += [radius]
            curve_fit_brightness_array += [brightness]
            curve_fit_brightness_err_array += [brightness_err]

        profile_sum = np.sum(on_profile_axis)
        start = (profile_sum, 0.5)
        popt, pcov = curve_fit(diffusion_func,curve_fit_radius_array,curve_fit_brightness_array,p0=start,sigma=curve_fit_brightness_err_array,absolute_sigma=True,bounds=((0, 0.01), (np.inf, np.inf)))
        profile_fit = diffusion_func(np.array(on_radial_axis), *popt)
        residual = np.array(on_profile_axis) - profile_fit
        chisq = np.sum((residual/np.array(on_profile_err_axis))**2)
        dof = len(on_radial_axis)-2
        print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print (f'plotname = {plotname}')
        print ('diffusion flux = %0.2E +/- %0.2E'%(popt[0],pow(pcov[0][0],0.5)))
        print ('diffusion radius = %0.2f +/- %0.2f deg (chi2/dof = %0.2f)'%(popt[1],pow(pcov[1][1],0.5),chisq/dof))

    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (f'plotname = {plotname}')
    lines = []
    lines += [f'Source location: RA = {roi_x:0.3f}, Dec = {roi_y:0.3f} deg \n']
    lines += [f'Energy range: {E_min:0.3f} - {E_max:0.3f} TeV \n']
    lines += ['radial distance [deg] \t surface brightness [TeV/cm2/s/sr] \t brightness error [TeV/cm2/s/sr] \n']
    for entry in range(0,len(on_radial_axis)):
        print (f'x = {on_radial_axis[entry]:0.2f}, y = {on_profile_axis[entry]:0.2e}, err = {on_profile_err_axis[entry]:0.2e}')
        lines += [f'{on_radial_axis[entry]:0.2f} \t {on_profile_axis[entry]:0.2e} \t {on_profile_err_axis[entry]:0.2e} \n']
    with open(f'output_plots/{plotname}_radial_profile.txt', 'w') as file:
        file.writelines(lines)

    uplims = np.zeros_like(on_radial_axis)
    #for x in range(0,len(on_radial_axis)):
    #    if on_profile_err_axis[x]==0.: continue
    #    significance = on_profile_axis[x]/on_profile_err_axis[x]
    #    if significance<2.:
    #        uplims[x] = 1.
    #        on_profile_axis[x] = max(2.*on_profile_err_axis[x],on_profile_axis[x]+2.*on_profile_err_axis[x])

    baseline_yaxis = [0. for i in range(0,len(on_radial_axis))]
    fig.clf()
    figsize_x = 7
    figsize_y = 5
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'angular distance [deg]'
    label_y = 'surface brightness [$\\mathrm{TeV}\\ \\mathrm{cm}^{-2}\\mathrm{s}^{-1}\\mathrm{sr}^{-1}$]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(on_radial_axis, baseline_yaxis, color='b', ls='dashed')
    #axbig.errorbar(on_radial_axis,on_profile_axis,on_profile_err_axis,uplims=uplims,color='k',marker='_',ls='none',zorder=2)
    axbig.errorbar(on_radial_axis,on_profile_axis,on_profile_err_axis,color='k',marker='_',ls='none',zorder=2)
    #axbig.fill_between(on_radial_axis,np.array(on_profile_axis)-np.array(profile_syst_err_axis),np.array(on_profile_axis)+np.array(profile_syst_err_axis),alpha=0.2,color='b',zorder=0)
    if fit_radial_profile:
        axbig.plot(on_radial_axis,diffusion_func(np.array(on_radial_axis),*popt),color='r')
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

def PrintSpectralDataForNaima(energy_axis,src_flux,src_flux_err,data_name):
    
    energy_mean_log = [] 
    energy_mean = [] 
    energy_edge_lo = [] 
    energy_edge_hi = [] 
    flux_mean = [] 
    flux_error = []
    ul = []
    for eb in range(0,len(energy_axis)):
        energy_mean_log += [math.log10(energy_axis[eb])]
    for eb in range(0,len(energy_axis)):
        energy_log_delta = 0.
        if eb+1<len(energy_axis):
            energy_log_delta = energy_mean_log[eb+1]-energy_mean_log[eb]
        else:
            energy_log_delta = energy_mean_log[eb]-energy_mean_log[eb-1]
        energy_mean += [pow(10,energy_mean_log[eb])]
        energy_edge_lo += [pow(10,energy_mean_log[eb]-0.5*energy_log_delta)]
        energy_edge_hi += [pow(10,energy_mean_log[eb]+0.5*energy_log_delta)]
        flux_mean += [src_flux[eb]/((energy_axis[eb])*(energy_axis[eb]))]
        flux_error += [src_flux_err[eb]/((energy_axis[eb])*(energy_axis[eb]))]
    print ('=======================================================')
    print ('NAIMA flux points')
    print ('data_name = %s'%(data_name))
    for eb in range(0,len(energy_axis)):
        print ('%.4f %.4f %.4f %.2e %.2e %s'%(energy_mean[eb],energy_edge_lo[eb],energy_edge_hi[eb],flux_mean[eb],flux_error[eb],0))
    print ('=======================================================')

    qfile = open("output_plots/naima_%s.dat"%(data_name),"w") 
    qfile.write("# %ECSV 0.9\n")
    qfile.write("# ---\n")
    qfile.write("# datatype:\n")
    qfile.write("# - {name: energy, unit: TeV, datatype: float64}\n")
    qfile.write("# - {name: energy_edge_lo, unit: TeV, datatype: float64}\n")
    qfile.write("# - {name: energy_edge_hi, unit: TeV, datatype: float64}\n")
    qfile.write("# - {name: flux, unit: 1 / (cm2 s TeV), datatype: float64}\n")
    qfile.write("# - {name: flux_error, unit: 1 / (cm2 s TeV), datatype: float64}\n")
    qfile.write("# - {name: ul, unit: '', datatype: int64}\n")
    qfile.write("# meta: !!omap\n")
    qfile.write("# - comments: [VHE gamma-ray spectrum of RX J1713.7-3946, 'Originally published in 2007\n")
    qfile.write("#       from 2003, 2004, and 2005 observations. The', spectrum here is as published\n")
    qfile.write("#       in the 2011 erratum, 'Main paper: Aharonian et al. 2007, A&A 464, 235', 'Erratum:\n")
    qfile.write("#       Aharonian et al. 2011, A&A 531, 1', Confidence level of upper limits is 2 sigma]\n")
    qfile.write("# - keywords: !!omap\n")
    qfile.write("#   - cl: {value: 0.95}\n")
    for eb in range(0,len(energy_axis)):
        qfile.write('%.2f %.2f %.2f %.2e %.2e %s\n'%(energy_mean[eb],energy_edge_lo[eb],energy_edge_hi[eb],flux_mean[eb],flux_error[eb],0))
    qfile.close() 

def ConvertRaDecToGalactic(ra, dec):
    my_sky = SkyCoord(ra*astropy_unit.deg, dec*astropy_unit.deg, frame='icrs')
    return my_sky.galactic.l.deg, my_sky.galactic.b.deg

    #delta = dec*np.pi/180.
    #delta_G = 27.12825*np.pi/180.
    #alpha = ra*np.pi/180.
    #alpha_G = 192.85948*np.pi/180.
    #l_NCP = 122.93192*np.pi/180.
    #sin_b = np.sin(delta)*np.sin(delta_G)+np.cos(delta)*np.cos(delta_G)*np.cos(alpha-alpha_G)
    #cos_b = np.cos(np.arcsin(sin_b))
    #sin_l_NCP_m_l = np.cos(delta)*np.sin(alpha-alpha_G)/cos_b
    #cos_l_NCP_m_l = (np.cos(delta_G)*np.sin(delta)-np.sin(delta_G)*np.cos(delta)*np.cos(alpha-alpha_G))/cos_b
    #b = (np.arcsin(sin_b))*180./np.pi
    #l = (l_NCP-np.arctan2(sin_l_NCP_m_l,cos_l_NCP_m_l))*180./np.pi
    #return l, b

def ConvertGalacticToRaDec(l, b):
    my_sky = SkyCoord(l*astropy_unit.deg, b*astropy_unit.deg, frame='galactic')
    return my_sky.icrs.ra.deg, my_sky.icrs.dec.deg

def GetSlicedDataCubeMap(map_file, sky_map, vel_low, vel_up):

    sky_map.reset()

    nbins_x = len(sky_map.xaxis)-1
    nbins_y = len(sky_map.yaxis)-1
    ra_min = sky_map.xaxis[0]
    ra_max = sky_map.xaxis[len(sky_map.xaxis)-1]
    dec_min = sky_map.yaxis[0]
    dec_max = sky_map.yaxis[len(sky_map.yaxis)-1]
    map_center_ra = 0.5 * (ra_min+ra_max)
    map_center_dec = 0.5 * (dec_min+dec_max)
    map_center_lon, map_center_lat = ConvertRaDecToGalactic(map_center_ra, map_center_dec)

    filename = map_file

    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)
    image_data = hdu.data
    print (f"image_data.shape = {image_data.shape}")
    print ("wcs")
    print (wcs)

    world_coord = wcs.all_pix2world(vel_low,map_center_lon,map_center_lat,1)
    min_vel = world_coord[0]
    world_coord = wcs.all_pix2world(vel_up,map_center_lon,map_center_lat,1)
    max_vel = world_coord[0]
    print (f"min_vel = {min_vel}")
    print (f"max_vel = {max_vel}")

    pixs_start = wcs.all_world2pix(vel_low,map_center_lon,map_center_lat,1)
    pixs_end = wcs.all_world2pix(vel_up,map_center_lon,map_center_lat,1)
    vel_idx_start = int(pixs_start[0])
    vel_idx_end = int(pixs_end[0])

    image_data_reduced_z = np.full((image_data[:, :, vel_idx_start].shape),0.)
    for idx in range(vel_idx_start,vel_idx_end):
        world_coord = wcs.all_pix2world(idx,0,0,1) 
        velocity = world_coord[0]
        world_coord = wcs.all_pix2world(idx+1,0,0,1) 
        velocity_next = world_coord[0]
        delta_vel = velocity_next - velocity
        image_data_reduced_z += image_data[:, :, idx]*delta_vel

    for idx_x in range(0,nbins_x):
        for idx_y in range(0,nbins_y):
            sky_ra = sky_map.xaxis[idx_x]
            sky_dec = sky_map.yaxis[idx_y]
            sky_lon, sky_lat = ConvertRaDecToGalactic(sky_ra, sky_dec)
            #if sky_lon>max_lon: continue
            #if sky_lat>max_lat: continue
            #if sky_lon<min_lon: continue
            #if sky_lat<min_lat: continue
            map_pixs = wcs.all_world2pix(vel_low, sky_lon, sky_lat, 1)
            pix_ra = int(map_pixs[1])
            pix_dec = int(map_pixs[2])
            #if pix_ra>=image_data_reduced_z.shape[0]: continue
            #if pix_dec>=image_data_reduced_z.shape[1]: continue
            sky_map.waxis[idx_x,idx_y,0] += image_data_reduced_z[pix_dec,pix_ra]

def GetSlicedDataCubeMapCGPS(map_file, sky_map, vel_low, vel_up):

    sky_map.reset()

    nbins_x = len(sky_map.xaxis)-1
    nbins_y = len(sky_map.yaxis)-1
    ra_min = sky_map.xaxis[0]
    ra_max = sky_map.xaxis[len(sky_map.xaxis)-1]
    dec_min = sky_map.yaxis[0]
    dec_max = sky_map.yaxis[len(sky_map.yaxis)-1]
    map_center_ra = 0.5 * (ra_min+ra_max)
    map_center_dec = 0.5 * (dec_min+dec_max)
    map_center_lon, map_center_lat = ConvertRaDecToGalactic(map_center_ra, map_center_dec)

    filename = map_file

    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)
    image_data = hdu.data
    print (f"image_data.shape = {image_data.shape}")
    print ("wcs")
    print (wcs)

    world_coord = wcs.all_pix2world(0,0,0,0,0)
    min_vel = world_coord[2]
    world_coord = wcs.all_pix2world(0,0,image_data.shape[1]-1,0,0)
    max_vel = world_coord[2]
    print (f"min_vel = {min_vel}")
    print (f"max_vel = {max_vel}")

    all_pix_lon = []
    all_pix_lat = []
    for idx_x in range(0,image_data.shape[2]-1):
        for idx_y in range(0,image_data.shape[3]-1):
            world_coord = wcs.all_pix2world(idx_x,idx_y,0,0,0) 
            lon = world_coord[0]
            lat = world_coord[1]
            all_pix_lon += [lon]
            all_pix_lat += [lat]
    max_lon = np.max(all_pix_lon)
    max_lat = np.max(all_pix_lat)
    min_lon = np.min(all_pix_lon)
    min_lat = np.min(all_pix_lat)
    print (f"max_lon = {max_lon}")
    print (f"max_lat = {max_lat}")
    print (f"min_lon = {min_lon}")
    print (f"min_lat = {min_lat}")

    pixs_start = wcs.all_world2pix(map_center_lon,map_center_lat,vel_up*1e3,1.0,1.0)
    pixs_end = wcs.all_world2pix(map_center_lon,map_center_lat,vel_low*1e3,1.0,1.0)
    print (f"pixs_start = {pixs_start}")
    print (f"pixs_end = {pixs_end}")
    vel_idx_start = int(pixs_start[2])
    vel_idx_end = int(pixs_end[2])

    image_data_reduced_z = np.full((image_data[0,vel_idx_start, :, :].shape),0.)
    for idx in range(vel_idx_start,vel_idx_end):
        world_coord = wcs.all_pix2world(0,0,idx,0,0) 
        velocity = world_coord[2]
        world_coord = wcs.all_pix2world(0,0,idx+1,0,0) 
        velocity_next = world_coord[2]
        delta_vel = velocity_next - velocity
        image_data_reduced_z += image_data[0,idx, :, :]

    for idx_x in range(0,nbins_x):
        for idx_y in range(0,nbins_y):
            sky_ra = sky_map.xaxis[idx_x]
            sky_dec = sky_map.yaxis[idx_y]
            sky_lon, sky_lat = ConvertRaDecToGalactic(sky_ra, sky_dec)
            if sky_lon>max_lon: continue
            if sky_lat>max_lat: continue
            if sky_lon<min_lon: continue
            if sky_lat<min_lat: continue
            map_pixs = wcs.all_world2pix(sky_lon, sky_lat, vel_low, 1, 1)
            pix_ra = int(map_pixs[0])
            pix_dec = int(map_pixs[1])
            if pix_ra>=image_data_reduced_z.shape[0]: continue
            if pix_dec>=image_data_reduced_z.shape[1]: continue
            sky_map.waxis[idx_x,idx_y,0] += image_data_reduced_z[pix_dec,pix_ra]


def GetSlicedDataCubeMapGALFA(map_file, sky_map, vel_low, vel_up):

    sky_map.reset()

    nbins_x = len(sky_map.xaxis)-1
    nbins_y = len(sky_map.yaxis)-1
    ra_min = sky_map.xaxis[0]
    ra_max = sky_map.xaxis[len(sky_map.xaxis)-1]
    dec_min = sky_map.yaxis[0]
    dec_max = sky_map.yaxis[len(sky_map.yaxis)-1]
    map_center_ra = 0.5 * (ra_min+ra_max)
    map_center_dec = 0.5 * (dec_min+dec_max)
    map_center_lon, map_center_lat = ConvertRaDecToGalactic(map_center_ra, map_center_dec)


    filename = map_file

    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)
    image_data = hdu.data
    print (f"image_data.shape = {image_data.shape}")
    print ("wcs")
    print (wcs)

    pixs_start = wcs.all_world2pix(map_center_ra,map_center_dec,vel_low,1)
    pixs_end = wcs.all_world2pix(map_center_ra,map_center_dec,vel_up,1)
    vel_idx_start = int(pixs_start[2])
    vel_idx_end = int(pixs_end[2])

    image_data_reduced_z = np.full((image_data[vel_idx_start, :, :].shape),0.)
    for idx in range(vel_idx_start,vel_idx_end):
        world_coord = wcs.all_pix2world(0,0,idx,1) 
        velocity = world_coord[2]
        world_coord = wcs.all_pix2world(0,0,idx+1,1) 
        velocity_next = world_coord[2]
        delta_vel = velocity_next - velocity
        image_data_reduced_z += image_data[idx, :, :]*delta_vel


    world_coord = wcs.all_pix2world(0,0,0,0)
    min_vel = world_coord[2]
    world_coord = wcs.all_pix2world(0,0,image_data.shape[1]-1,0)
    max_vel = world_coord[2]
    print (f"min_vel = {min_vel}")
    print (f"max_vel = {max_vel}")

    all_pix_lon = []
    all_pix_lat = []
    for idx_x in range(0,image_data.shape[1]-1):
        for idx_y in range(0,image_data.shape[2]-1):
            world_coord = wcs.all_pix2world(idx_x,idx_y,0,0) 
            lon = world_coord[0]
            lat = world_coord[1]
            all_pix_lon += [lon]
            all_pix_lat += [lat]
    max_lon = np.max(all_pix_lon)
    max_lat = np.max(all_pix_lat)
    min_lon = np.min(all_pix_lon)
    min_lat = np.min(all_pix_lat)
    print (f"max_lon = {max_lon}")
    print (f"max_lat = {max_lat}")
    print (f"min_lon = {min_lon}")
    print (f"min_lat = {min_lat}")

    for idx_x in range(0,nbins_x):
        for idx_y in range(0,nbins_y):
            sky_ra = sky_map.xaxis[idx_x]
            sky_dec = sky_map.yaxis[idx_y]
            map_pixs = wcs.all_world2pix(sky_ra, sky_dec, vel_low, 1)
            pix_ra = int(map_pixs[0])
            pix_dec = int(map_pixs[1])
            if pix_ra>=image_data_reduced_z.shape[1]: continue
            if pix_dec>=image_data_reduced_z.shape[0]: continue
            sky_map.waxis[idx_x,idx_y,0] += image_data_reduced_z[pix_dec,pix_ra]

def compute_camera_frame_power_spectrum(skymap,idx_z=0):

    nbins_x = len(skymap.xaxis)-1
    nbins_y = len(skymap.yaxis)-1

    data = []
    for idx_x in range(0,nbins_x):
        data_y = []
        for idx_y in range(0,nbins_y):
            data_y += [skymap.waxis[idx_x,idx_y,idx_z]]
        data += [data_y]
    data = np.array(data)

    rng = np.random.default_rng()
    noise = np.zeros_like(data)
    for idx_x in range(0,nbins_x):
        for idx_y in range(0,nbins_y):
            if data[idx_x][idx_y]==0.:
                continue
            noise[idx_x][idx_y] = rng.standard_normal()

    # Compute the 2D Fourier Transform
    # This function takes a 2D array as input and returns its 2D Fourier Transform, which is also a 2D array of complex numbers.
    fourier_transform = np.fft.fft2(data)
    fourier_transform_noise = np.fft.fft2(noise)

    magnitude_noise = np.abs(fourier_transform_noise)
    magnitude = np.abs(fourier_transform)
    phase = np.angle(fourier_transform)

    v_power_spectrum = [0.] * magnitude.shape[0]
    h_power_spectrum = [0.] * magnitude.shape[1]
    v_power_spectrum_noise = [0.] * magnitude.shape[0]
    h_power_spectrum_noise = [0.] * magnitude.shape[1]
    for idx_x in range(0,magnitude.shape[0]):
        h_power_spectrum[idx_x] += np.sum(magnitude[idx_x,:])
        h_power_spectrum_noise[idx_x] += np.sum(magnitude_noise[idx_x,:])
    for idx_y in range(0,magnitude.shape[1]):
        v_power_spectrum[idx_y] += np.sum(magnitude[:,idx_y])
        v_power_spectrum_noise[idx_y] += np.sum(magnitude_noise[:,idx_y])

    # Shift the zero-frequency component to the center
    v_power_spectrum_shifted = np.abs(np.fft.fftshift(v_power_spectrum))
    h_power_spectrum_shifted = np.abs(np.fft.fftshift(h_power_spectrum))
    v_power_spectrum_noise_shifted = np.abs(np.fft.fftshift(v_power_spectrum_noise))
    h_power_spectrum_noise_shifted = np.abs(np.fft.fftshift(h_power_spectrum_noise))

    for idx_x in range(0,len(v_power_spectrum_noise_shifted)):
        if v_power_spectrum_noise_shifted[idx_x]==0.:
            v_power_spectrum_shifted[idx_x] = 0.
        else:
            v_power_spectrum_shifted[idx_x] = (v_power_spectrum_shifted[idx_x]-v_power_spectrum_noise_shifted[idx_x])/v_power_spectrum_noise_shifted[idx_x]
    for idx_y in range(0,len(h_power_spectrum_noise_shifted)):
        if h_power_spectrum_noise_shifted[idx_y]==0.:
            h_power_spectrum_shifted[idx_y] = 0.
        else:
            h_power_spectrum_shifted[idx_y] = (h_power_spectrum_shifted[idx_y]-h_power_spectrum_noise_shifted[idx_y])/h_power_spectrum_noise_shifted[idx_y]

    # Calculate the corresponding frequencies
    freqs = np.fft.fftfreq(nbins_y, skymap.yaxis[1]-skymap.yaxis[0])
    freqs_shifted = np.fft.fftshift(freqs)

    return freqs_shifted, v_power_spectrum_shifted, h_power_spectrum_shifted

    #v_size = magnitude.shape[0]//2+1
    #h_size = magnitude.shape[1]//2+1
    #rv_power_spectrum = [0.] * v_size
    #rh_power_spectrum = [0.] * h_size
    #rv_power_spectrum[0] = v_power_spectrum_shifted[v_size-1]
    #rh_power_spectrum[0] = v_power_spectrum_shifted[h_size-1]
    #for idx_x in range(1,v_size-1):
    #    rv_power_spectrum[idx_x] += v_power_spectrum_shifted[v_size-1+idx_x]
    #    rv_power_spectrum[idx_x] += v_power_spectrum_shifted[v_size-1-idx_x]
    #for idx_y in range(1,h_size-1):
    #    rh_power_spectrum[idx_y] += h_power_spectrum_shifted[h_size-1+idx_y]
    #    rh_power_spectrum[idx_y] += h_power_spectrum_shifted[h_size-1-idx_y]

    #freqs = np.fft.rfftfreq(nbins_y, skymap.yaxis[1]-skymap.yaxis[0])

    #return freqs, np.array(rv_power_spectrum), np.array(rh_power_spectrum)

def plot_camera_frame_power_spectrum(fig,plotname,skymap,idx_z=0):

    freqs_shifted, v_power_spectrum, h_power_spectrum = compute_camera_frame_power_spectrum(skymap,idx_z=idx_z)
    fig.clf()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'wave number $k$'
    label_y = 'power spectrum (vertical)'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(freqs_shifted,v_power_spectrum)
    fig.savefig(f'output_plots/power_spectrum_v_{plotname}.png',bbox_inches='tight')
    axbig.remove()
    fig.clf()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'wave number $k$'
    label_y = 'power spectrum (horizontal)'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(freqs_shifted,h_power_spectrum)
    fig.savefig(f'output_plots/power_spectrum_h_{plotname}.png',bbox_inches='tight')
    axbig.remove()

def build_skymap(
        source_name,
        src_ra,
        src_dec,
        smi_input,
        neuralnet_path,
        eigenvector_path,
        big_matrix_path,
        runlist,
        off_runlist,
        mimic_runlist,
        onoff, 
        incl_sky_map, 
        data_sky_map, 
        fit_sky_map, 
        syst_sky_map, 
        data_xyoff_map, 
        fit_xyoff_map, 
        init_xyoff_map, 
        data_xyvar_map, 
        syst_xyoff_map,
        total_data_sky_map,
        total_bkgd_sky_map,
    ):

    global skymap_bins
    if onoff=='ON' or 'MIMIC' in onoff:
        skymap_bins = fine_skymap_bins

    # start memory profiling
    tracemalloc.start()


    print ('loading neural net... ')
    nn_model = pickle.load(open(neuralnet_path, "rb"))

    print ('loading svd pickle data... ')
    input_filename = eigenvector_path
    eigen_stuff = pickle.load(open(input_filename, "rb"))
    big_xyoff_eigenvalues_fullspec = eigen_stuff[0][0]
    big_xyoff_eigenvectors_fullspec = eigen_stuff[0][1]
    avg_xyoff_map_1d_fullspec = eigen_stuff[0][2]


    exposure_hours = 0.
    avg_tel_elev = 0.
    avg_tel_azim = 0.
    avg_MeanPedvar = 0.
    total_events = 0.

    print ('build big matrix...')

    big_on_matrix_fullspec = []
    big_off_matrix_fullspec = []
    big_mask_matrix_fullspec = []
    big_off_mask_matrix_fullspec = []

    print (f'runlist = {runlist}')
    print (f'off_runlist = {off_runlist}')
    print (f'mimic_runlist = {mimic_runlist}')

    if 'MIMIC' in onoff:
        mimic_index = int(onoff.strip('MIMIC'))-1
        new_mimic_runlist = []
        for run in range(0,len(runlist)):
            if run>=len(mimic_runlist): continue
            if mimic_index>=len(mimic_runlist[run]): continue
            new_mimic_runlist += [mimic_runlist[run][mimic_index]]
        big_on_elevation, big_on_exposure, big_on_matrix_fullspec, big_mask_matrix_fullspec = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,new_mimic_runlist,max_runs=1e10,is_bkgd=False)
    elif 'ON' in onoff:
        big_on_elevation, big_on_exposure, big_on_matrix_fullspec, big_mask_matrix_fullspec = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,runlist,max_runs=1e10,is_bkgd=False)
    else:
        big_on_elevation, big_on_exposure, big_on_matrix_fullspec, big_mask_matrix_fullspec = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,runlist,max_runs=1e10,is_bkgd=False)
    #big_off_elevation, big_off_exposure, big_off_matrix_fullspec, big_off_mask_matrix_fullspec = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,off_runlist,max_runs=1e10,is_bkgd=True)

    if len(big_on_matrix_fullspec)==0:
        print (f'No data. Break.')
        return [exposure_hours,avg_tel_elev,avg_tel_azim,avg_MeanPedvar]

    ratio_xyoff_map = []
    for logE in range(0,logE_nbins):
        ratio_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    
    for logE in range(0,logE_nbins):
        incl_sky_map[logE].reset()
        data_sky_map[logE].reset()
        fit_sky_map[logE].reset()
        syst_sky_map[logE].reset()
        data_xyoff_map[logE].reset()
        fit_xyoff_map[logE].reset()
        init_xyoff_map[logE].reset()
        data_xyvar_map[logE].reset()
        syst_xyoff_map[logE].reset()
        ratio_xyoff_map[logE].reset()

    print ('===================================================================================')
    print ('fitting xyoff maps...')

    data_multivar_map_1d_fullspec = np.zeros_like(big_on_matrix_fullspec[0])
    mask_multivar_map_1d_fullspec = np.zeros_like(big_mask_matrix_fullspec[0])
    for entry in range(0,len(big_on_matrix_fullspec)):
        data_multivar_map_1d_fullspec += np.array(big_on_matrix_fullspec[entry])
        mask_multivar_map_1d_fullspec += np.array(big_mask_matrix_fullspec[entry])

    data_xyoff_map_1d_fullspec = convert_multivar_to_xyoff_vector1d(data_multivar_map_1d_fullspec)
    mask_xyoff_map_1d_fullspec = convert_multivar_to_xyoff_vector1d(mask_multivar_map_1d_fullspec)
    data_xyvar_map_1d_fullspec = convert_multivar_to_xyvar_vector1d(data_multivar_map_1d_fullspec)
    mask_xyvar_map_1d_fullspec = convert_multivar_to_xyvar_vector1d(mask_multivar_map_1d_fullspec)

    #off_data_multivar_map_1d_fullspec = np.zeros_like(big_off_matrix_fullspec[0])
    #off_mask_multivar_map_1d_fullspec = np.zeros_like(big_off_mask_matrix_fullspec[0])
    #for entry in range(0,len(big_off_matrix_fullspec)):
    #    off_data_multivar_map_1d_fullspec += np.array(big_off_matrix_fullspec[entry])
    #    off_mask_multivar_map_1d_fullspec += np.array(big_off_mask_matrix_fullspec[entry])

    #off_data_xyoff_map_1d_fullspec = convert_multivar_to_xyoff_vector1d(off_data_multivar_map_1d_fullspec)
    #off_mask_xyoff_map_1d_fullspec = convert_multivar_to_xyoff_vector1d(off_mask_multivar_map_1d_fullspec)
    #off_data_xyvar_map_1d_fullspec = convert_multivar_to_xyvar_vector1d(off_data_multivar_map_1d_fullspec)
    #off_mask_xyvar_map_1d_fullspec = convert_multivar_to_xyvar_vector1d(off_mask_multivar_map_1d_fullspec)

    xyoff_idx_1d = find_index_for_xyoff_vector1d()
    xyvar_idx_1d = find_index_for_xyvar_vector1d()
    for logE in range(0,logE_nbins):
        for gcut in range(0,gcut_bins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d = xyoff_idx_1d[gcut][logE][idx_x][idx_y]
                    data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = data_xyoff_map_1d_fullspec[idx_1d]
                    #init_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = off_data_xyoff_map_1d_fullspec[idx_1d]
                    #fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = off_data_xyoff_map_1d_fullspec[idx_1d]
                    init_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = avg_xyoff_map_1d_fullspec[idx_1d]
                    fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = avg_xyoff_map_1d_fullspec[idx_1d]

    for logE in range(0,logE_nbins):
        for idx_x in range(0,xvar_bins[logE]):
            for idx_y in range(0,yvar_bins[logE]):
                idx_1d = xyvar_idx_1d[logE][idx_x][idx_y]
                data_xyvar_map[logE].waxis[idx_x,idx_y,0] = data_xyvar_map_1d_fullspec[idx_1d]

    sr_map_1d_truth, cr_map_1d = prepare_vector_for_least_square(data_multivar_map_1d_fullspec)
    ls_model = nn_model[0]
    ls_model_err = nn_model[1]
    sr_map_1d = np.zeros_like(sr_map_1d_truth)
    sr_map_1d_err = np.zeros_like(sr_map_1d_truth)
    for logE in range(0,logE_nbins):
        sr_map_1d[logE] = cr_map_1d @ ls_model[logE]
        sr_map_1d_err[logE] = cr_map_1d @ ls_model_err[logE]

    for logE in range(0,logE_nbins):
        norm_cr_data = np.sum(data_xyoff_map[logE].waxis[:,:,:]) - np.sum(data_xyoff_map[logE].waxis[:,:,0])
        norm_sr_data = sr_map_1d[logE]
        norm_sr_data_err = sr_map_1d_err[logE]
        norm_sr_init = np.sum(init_xyoff_map[logE].waxis[:,:,0])
        norm_cr_init = np.sum(init_xyoff_map[logE].waxis[:,:,:]) - np.sum(init_xyoff_map[logE].waxis[:,:,0])
        norm_data = norm_cr_data
        norm_init = norm_cr_init
        if abs(norm_sr_data_err)<abs(norm_sr_data):
            #norm_data = norm_sr_data + norm_cr_data
            #norm_init = norm_sr_init + norm_cr_init
            norm_data = norm_sr_data 
            norm_init = norm_sr_init 
        if norm_init==0.: continue
        for gcut in range(0,gcut_bins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    init_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = init_xyoff_map[logE].waxis[idx_x,idx_y,gcut] * norm_data/norm_init
                    fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] * norm_data/norm_init

    logE_peak = 0
    bkgd_peak = 0.
    for logE in range(0,logE_nbins):
        bkgd = np.sum(fit_xyoff_map[logE].waxis[:,:,:])
        if bkgd>bkgd_peak:
            bkgd_peak = bkgd
            logE_peak = logE
    print (f'logE_peak = {logE_peak}')

    if not use_init:

        print ('===================================================================================')
        print ('fitting xyoff maps fullspec...')

        xyoff_idx_1d = find_index_for_xyoff_vector1d()
        init_xyoff_map_1d_fullspec = np.zeros_like(data_xyoff_map_1d_fullspec)
        syst_xyoff_map_1d_fullspec = np.zeros_like(data_xyoff_map_1d_fullspec)
        for logE in range(0,logE_nbins):
            for gcut in range(0,gcut_bins):
                norm_init = np.sum(init_xyoff_map[logE].waxis[:,:,gcut])
                for idx_x in range(0,xoff_bins[logE]):
                    for idx_y in range(0,yoff_bins[logE]):
                        idx_1d = xyoff_idx_1d[gcut][logE][idx_x][idx_y]
                        prediction = init_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
                        init_xyoff_map_1d_fullspec[idx_1d] = prediction
                        rel_syst = 1.
                        if abs(sr_map_1d[logE])>0.:
                            rel_syst = sr_map_1d_err[logE]/abs(sr_map_1d[logE])
                        syst_xyoff_map_1d_fullspec[idx_1d] = prediction * rel_syst

        for logE in range(0,logE_nbins):
            for gcut in range(0,gcut_bins):
                for idx_x in range(0,xoff_bins[logE]):
                    for idx_y in range(0,yoff_bins[logE]):
                        idx_1d = xyoff_idx_1d[gcut][logE][idx_x][idx_y]
                        prediction = max(0.0,init_xyoff_map_1d_fullspec[idx_1d])
                        syst_error = abs(syst_xyoff_map_1d_fullspec[idx_1d])
                        fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = prediction
                        syst_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = syst_error

        for logE in range(0,logE_nbins):
            truth = np.sum(data_xyoff_map[logE].waxis[:,:,0])
            prediction = sr_map_1d[logE]
            error = sr_map_1d_err[logE]
            print (f"truth = {truth:0.1f}, prediction = {prediction:0.1f}+/-{error:0.1f}")


        xyoff_effective_matrix_rank_fullspec = big_xyoff_eigenvectors_fullspec.shape[0]

        xyoff_truth_params = big_xyoff_eigenvectors_fullspec @ data_xyoff_map_1d_fullspec
        xyoff_avg_params = big_xyoff_eigenvectors_fullspec @ init_xyoff_map_1d_fullspec
        xyoff_fit_params = big_xyoff_eigenvectors_fullspec @ init_xyoff_map_1d_fullspec

        init_params = xyoff_fit_params
        stepsize = [1e-4] * xyoff_effective_matrix_rank_fullspec
        solution = minimize(
            cosmic_ray_like_chi2_fullspec,
            x0=init_params,
            args=(
                big_xyoff_eigenvectors_fullspec,
                data_xyoff_map_1d_fullspec,
                mask_xyoff_map_1d_fullspec,
                init_xyoff_map_1d_fullspec,
                syst_xyoff_map_1d_fullspec,
                True,
            ),
            method='L-BFGS-B',
            jac=None,
            options={'eps':stepsize,'ftol':0.0001},
        )
        print (f"solution['fun'] = {solution['fun']}")
        fit_params = solution['x']

        print ("***************************************************************************************")
        for entry in range(0,len(fit_params)):
            print (f"init_params = {init_params[entry]:0.1f}, fit_params = {fit_params[entry]:0.1f}")
            if np.isnan(fit_params[entry]):
                print ("Soluiton is nan!!!")
                #exit()
                for logE in range(0,logE_nbins):
                    incl_sky_map[logE].reset()
                    data_sky_map[logE].reset()
                    fit_sky_map[logE].reset()
                    syst_sky_map[logE].reset()
                    data_xyoff_map[logE].reset()
                    fit_xyoff_map[logE].reset()
                    init_xyoff_map[logE].reset()
                    data_xyvar_map[logE].reset()
                    syst_xyoff_map[logE].reset()
                    ratio_xyoff_map[logE].reset()
                return [exposure_hours,avg_tel_elev,avg_tel_azim,avg_MeanPedvar]

        fit_xyoff_map_1d_fullspec = big_xyoff_eigenvectors_fullspec.T @ fit_params

        xyoff_idx_1d = find_index_for_xyoff_vector1d()
        for logE in range(0,logE_nbins):
            for gcut in range(0,gcut_bins):
                for idx_x in range(0,xoff_bins[logE]):
                    for idx_y in range(0,yoff_bins[logE]):
                        idx_1d = xyoff_idx_1d[gcut][logE][idx_x][idx_y]
                        prediction = max(0.0,fit_xyoff_map_1d_fullspec[idx_1d])
                        fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = prediction




    print ('===================================================================================')
    for logE in range(0,logE_nbins):
        print (f'E = {pow(10.,logE_bins[logE]):0.3f}')
        for gcut in range(0,gcut_bins):
            if gcut!=0: continue
            sum_data_xyoff_map = np.sum(data_xyoff_map[logE].waxis[:,:,gcut])
            sum_init_xyoff_map = np.sum(init_xyoff_map[logE].waxis[:,:,gcut])
            sum_fit_xyoff_map = np.sum(fit_xyoff_map[logE].waxis[:,:,gcut])
            print (f'sum_data_xyoff_map = {sum_data_xyoff_map:0.1f}, sum_init_xyoff_map = {sum_init_xyoff_map:0.1f}, sum_fit_xyoff_map = {sum_fit_xyoff_map:0.1f}')


    for logE in range(0,logE_nbins):
        for idx_x in range(0,xoff_bins[logE]):
            for idx_y in range(0,yoff_bins[logE]):
                sum_xyoff_map_cr = 0.
                for gcut in range(1,3):
                    model = fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
                    sum_xyoff_map_cr += model
                for gcut in range(0,gcut_bins):
                    model = fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
                    model_err = syst_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
                    ratio = 0.
                    ratio_syst = 0.
                    if sum_xyoff_map_cr>0.:
                        ratio = model/sum_xyoff_map_cr
                        ratio_syst = model_err/sum_xyoff_map_cr
                    ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = ratio
                    syst_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = ratio_syst


    for logE in range(0,logE_nbins):

        if xoff_bins[logE]==1: continue
        if yoff_bins[logE]==1: continue

        sum_xyoff_map_sr = 0.
        sum_xyoff_map_sr = np.sum(fit_xyoff_map[logE].waxis[:,:,0])
        sum_xyoff_map_cr = 0.
        for gcut in range(1,3):
            sum_xyoff_map_cr += np.sum(data_xyoff_map[logE].waxis[:,:,gcut])
        avg_ratio = 0.
        if sum_xyoff_map_cr>0.:
            avg_ratio = sum_xyoff_map_sr/sum_xyoff_map_cr

        if sum_xyoff_map_sr<float(xoff_bins[logE]*yoff_bins[logE]):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    ratio_xyoff_map[logE].waxis[idx_x,idx_y,0] = avg_ratio




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
            if mimic_index+1>len(mimic_runlist[run]):
                print (f'Not enough mimic data.')
                continue
            run_number = mimic_runlist[run][mimic_index]
            print (f'analyzing mimic run {run_number}')

            InputFile.Close()
            rootfile_name = f'{smi_input}/{run_number}.anasum.root'
            print (rootfile_name)
            if not os.path.exists(rootfile_name):
                print (f'file does not exist.')
                continue
            InputFile = ROOT.TFile(rootfile_name)

        list_timecuts = GetRunTimecuts(int(run_number))
        print (f"run_number = {run_number}, list_timecuts = {list_timecuts}")

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
            RA = EvtTree.RA
            DEC = EvtTree.DEC
            Xoff = EvtTree.Xoff
            Yoff = EvtTree.Yoff
            Xderot = EvtTree.Xderot
            Yderot = EvtTree.Yderot
            Energy = EvtTree.Energy
            NImages = EvtTree.NImages
            EmissionHeight = EvtTree.EmissionHeight
            MeanPedvar = EvtTree.MeanPedvar
            Xcore = EvtTree.XCore
            Ycore = EvtTree.YCore
            Time = EvtTree.timeOfDay
            Roff = pow(Xoff*Xoff+Yoff*Yoff,0.5)
            Rcore = pow(Xcore*Xcore+Ycore*Ycore,0.5)
            logE = logE_axis.get_bin(np.log10(Energy))

            if NImages<2: continue
            if not ApplyTimeCuts(Time-time_start,list_timecuts): continue
            if logE<0: continue
            if logE>=len(data_sky_map): continue
            if Energy<min_Energy_cut: continue
            if Energy>max_Energy_cut: continue

            MSCW = EvtTree.MSCW/MSCW_cut[logE]
            MSCL = EvtTree.MSCL/MSCL_cut[logE]
            GammaCut = EventGammaCut(MSCL,MSCW)
            if GammaCut>float(gcut_end): continue

            if NImages<min_NImages: continue
            if EmissionHeight>max_EmissionHeight_cut: continue
            if EmissionHeight<min_EmissionHeight_cut: continue
            if MeanPedvar>max_MeanPedvar_cut: continue
            if MeanPedvar<min_MeanPedvar_cut: continue
            if Roff>max_Roff: continue
            if Rcore>max_Rcore: continue
            if Rcore<min_Rcore: continue

            avg_MeanPedvar += MeanPedvar
            total_events += 1.

            Xsky = RA
            Ysky = DEC
            if 'MIMIC' in onoff:
                Xsky = TelRAJ2000 + Xderot
                Ysky = TelDecJ2000 + Yderot

            if onoff=='OFF':
                found_gamma_source = CoincideWithBrightStars(Xsky, Ysky, gamma_source_coord)
                if found_gamma_source: continue
            if 'MIMIC' in onoff:
                found_gamma_source = CoincideWithBrightStars(RA, DEC, gamma_source_coord)
                if found_gamma_source: continue

            if onoff=='OFF':

                Xsky_rel = RA - src_ra
                Ysky_rel = DEC - src_dec

                incl_sky_map[logE].fill(Xsky_rel,Ysky_rel,0.5)
                if GammaCut>float(gcut_end): continue

                sr_syst = syst_xyoff_map[logE].get_bin_content(Xoff,Yoff,0.5)
                sr_model = ratio_xyoff_map[logE].get_bin_content(Xoff,Yoff,0.5)
                if GammaCut>1. and GammaCut<3.:
                    fit_sky_map[logE].fill(Xsky_rel,Ysky_rel,0.5,weight=sr_model)
                    syst_sky_map[logE].fill(Xsky_rel,Ysky_rel,0.5,weight=sr_syst)
                elif GammaCut<1.:
                    data_sky_map[logE].fill(Xsky_rel,Ysky_rel,GammaCut)

            else:
                if coordinate_type == 'galactic':

                    Gal_Xsky, Gal_Ysky = ConvertRaDecToGalactic(Xsky, Ysky)

                    incl_sky_map[logE].fill(Gal_Xsky,Gal_Ysky,0.5)
                    if GammaCut>float(gcut_end): continue

                    sr_syst = syst_xyoff_map[logE].get_bin_content(Xoff,Yoff,0.5)
                    sr_model = ratio_xyoff_map[logE].get_bin_content(Xoff,Yoff,0.5)
                    if GammaCut>1. and GammaCut<3.:
                        fit_sky_map[logE].fill(Gal_Xsky,Gal_Ysky,0.5,weight=sr_model)
                        syst_sky_map[logE].fill(Gal_Xsky,Gal_Ysky,0.5,weight=sr_syst)
                    elif GammaCut<1.:
                        data_sky_map[logE].fill(Gal_Xsky,Gal_Ysky,GammaCut)

                else:

                    incl_sky_map[logE].fill(Xsky,Ysky,0.5)
                    if GammaCut>float(gcut_end): continue

                    sr_syst = syst_xyoff_map[logE].get_bin_content(Xoff,Yoff,0.5)
                    sr_model = ratio_xyoff_map[logE].get_bin_content(Xoff,Yoff,0.5)
                    if GammaCut>1. and GammaCut<3.:
                        fit_sky_map[logE].fill(Xsky,Ysky,0.5,weight=sr_model)
                        syst_sky_map[logE].fill(Xsky,Ysky,0.5,weight=sr_syst)
                    elif GammaCut<1.:
                        data_sky_map[logE].fill(Xsky,Ysky,GammaCut)
    
        print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')

        InputFile.Close()
  
    tracemalloc.stop()
    if exposure_hours>0.:
        avg_tel_elev = avg_tel_elev/exposure_hours
        avg_tel_azim = avg_tel_azim/exposure_hours
        avg_MeanPedvar = avg_MeanPedvar/total_events

    print (f'avg_tel_elev = {avg_tel_elev}')

    del big_on_matrix_fullspec
    del big_off_matrix_fullspec
    del big_mask_matrix_fullspec
    del big_off_mask_matrix_fullspec

    return [exposure_hours,avg_tel_elev,avg_tel_azim,avg_MeanPedvar]

