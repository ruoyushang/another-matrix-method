
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from common_functions import MyArray1D
from common_functions import MyArray3D

import common_functions

#logE_min = common_functions.logE_min
#logE_mid = common_functions.logE_mid
#logE_max = common_functions.logE_max
logE_axis = common_functions.logE_axis
logE_nbins = common_functions.logE_nbins
logE_bins = common_functions.logE_bins
gcut_bins = common_functions.gcut_bins
xoff_bins = common_functions.xoff_bins
yoff_bins = common_functions.yoff_bins
xoff_start = common_functions.xoff_start
xoff_end = common_functions.xoff_end
yoff_start = common_functions.yoff_start
yoff_end = common_functions.yoff_end
gcut_start = common_functions.gcut_start
gcut_end = common_functions.gcut_end
SaveFITS = common_functions.SaveFITS
GetSlicedDataCubeMap = common_functions.GetSlicedDataCubeMap
GetSlicedDataCubeMapGALFA = common_functions.GetSlicedDataCubeMapGALFA
ReadRunListFromFile = common_functions.ReadRunListFromFile
build_skymap = common_functions.build_skymap
smooth_image = common_functions.smooth_image
PlotSkyMap = common_functions.PlotSkyMap
PlotCountProjection = common_functions.PlotCountProjection
make_flux_map = common_functions.make_flux_map
make_significance_map = common_functions.make_significance_map
DefineRegionOfInterest = common_functions.DefineRegionOfInterest
PrintAndPlotInformationRoI = common_functions.PrintAndPlotInformationRoI
GetRadialProfile = common_functions.GetRadialProfile
fit_2d_model = common_functions.fit_2d_model
matrix_rank = common_functions.matrix_rank
skymap_size = common_functions.skymap_size
skymap_bins = common_functions.skymap_bins
fine_skymap_bins = common_functions.fine_skymap_bins
GetGammaSourceInfo = common_functions.GetGammaSourceInfo
build_radial_symmetric_model = common_functions.build_radial_symmetric_model
doFluxCalibration = common_functions.doFluxCalibration
diffusion_func = common_functions.diffusion_func
significance_li_and_ma = common_functions.significance_li_and_ma


fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_dir = os.environ.get("SMI_DIR")
smi_input = os.environ.get("SMI_INPUT")
#smi_output = os.environ.get("SMI_OUTPUT")
#smi_output = "/nevis/ged/data/rshang/smi_output/output_default"
smi_output = "/nevis/ged/data/rshang/smi_output/output_7x7"
#smi_output = "/nevis/ged/data/rshang/smi_output/output_detail"

smooth_size = 0.06
#smooth_size = 0.08
#smooth_size = 0.2

zoomin = 1.0
#zoomin = 0.5

#ana_tag = 'demo'
#ana_tag = 'linear'
#ana_tag = 'poisson'
#ana_tag = 'binspec'
#ana_tag = 'fullspec'
#ana_tag = 'init'
#ana_tag = 'rank1'
#ana_tag = 'rank2'
ana_tag = 'rank4'
#ana_tag = 'rank8'
#ana_tag = 'rank16'
#ana_tag = 'rank32'
#ana_tag = 'rank64'

qual_cut = 0.
#qual_cut = 20.

elev_cut = 20.
#elev_cut = 55.
cr_qual_cut = 1e10

#bias_array = [-0.023, -0.011, -0.022, -0.03,  -0.025,  0.018, -0.048, -0.378, -0.271]

if ana_tag=='demo':
    xoff_bins = [11,11,11,11,11,11,11,11]
    yoff_bins = xoff_bins

source_name = sys.argv[1]
src_ra = float(sys.argv[2])
src_dec = float(sys.argv[3])
onoff = sys.argv[4]

n_mimic = 0
if onoff=='ON':
    n_mimic = 0
    #n_mimic = 5

#input_epoch = ['V4']
#input_epoch = ['V5']
#input_epoch = ['V6']
input_epoch = ['V4','V5','V6']

logE_min = 0
logE_mid = 4
logE_max = logE_nbins
fit_radial_profile = False
radial_bin_scale = 0.1

#include_syst_error = True
include_syst_error = False

if 'Crab' in source_name:
    logE_min = 0
    logE_mid = 4
    logE_max = logE_nbins
    fit_radial_profile = False
if 'SNR_G189_p03' in source_name:
    logE_min = 1
    logE_mid = 6
    logE_max = logE_nbins
    fit_radial_profile = False
    radial_bin_scale = 0.3
if 'PSR_J1856_p0245' in source_name:
    #logE_min = 2
    #logE_mid = 5
    logE_min = 0
    logE_mid = 4
    logE_max = logE_nbins
    fit_radial_profile = False
if 'PSR_J1907_p0602' in source_name:
    logE_min = 3
    logE_mid = 5
    #logE_min = 0
    #logE_mid = 4
    logE_max = logE_nbins
    fit_radial_profile = False
if 'SS433' in source_name:
    logE_min = 0
    logE_mid = 4
    logE_max = logE_nbins
    fit_radial_profile = False
if 'PSR_J2021_p4026' in source_name:
    #logE_min = 2
    #logE_mid = 4
    logE_min = 0
    logE_mid = 3
    logE_max = logE_nbins
    fit_radial_profile = False
    radial_bin_scale = 0.2
if 'PSR_J2229_p6114' in source_name:
    logE_min = 0
    logE_mid = 4
    logE_max = logE_nbins
    fit_radial_profile = False
if 'Geminga' in source_name:
    logE_min = 0
    logE_mid = 5
    logE_max = logE_nbins
    fit_radial_profile = True
    radial_bin_scale = 0.25

if doFluxCalibration:
    logE_min = 0
    logE_mid = 4
    logE_max = logE_nbins

xsky_start = src_ra+skymap_size
xsky_end = src_ra-skymap_size
ysky_start = src_dec-skymap_size
ysky_end = src_dec+skymap_size

if onoff=='ON':
    skymap_bins = fine_skymap_bins
    print (f'original skymap_bins = {skymap_bins}')
    skymap_bin_size = 2.*skymap_size/float(skymap_bins)

    #skymap_size = 1.
    #xsky_start = src_ra+skymap_size
    #xsky_end = src_ra-skymap_size
    #ysky_start = src_dec-skymap_size
    #ysky_end = src_dec+skymap_size
    #skymap_bins = int(2.*skymap_size/skymap_bin_size)
    #print (f'final skymap_bins = {skymap_bins}')

region_name = source_name
if onoff=='OFF':
    region_name = 'Validation'
all_roi_name, all_roi_x, all_roi_y, all_roi_r, all_excl_x, all_excl_y, all_excl_r = DefineRegionOfInterest(region_name,src_ra,src_dec)

total_exposure = 0.
good_exposure = 0.
mimic_exposure = [0.] * n_mimic 
list_run_elev = []
list_run_azim = []
list_truth_params = []
list_fit_params = []
sum_incl_sky_map = []
sum_data_sky_map = []
sum_bkgd_sky_map = []
sum_syst_sky_map = []
sum_incl_sky_map_smooth = []
sum_data_sky_map_smooth = []
sum_bkgd_sky_map_smooth = []
sum_syst_sky_map_smooth = []
sum_excess_sky_map_smooth = []
sum_significance_sky_map = []
sum_flux_sky_map = []
sum_flux_err_sky_map = []
sum_flux_syst_sky_map = []
sum_flux_sky_map_smooth = []
sum_flux_err_sky_map_smooth = []
sum_flux_syst_sky_map_smooth = []
for logE in range(0,logE_nbins):
    sum_incl_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_syst_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_incl_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_data_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_bkgd_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_syst_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_excess_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_significance_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_err_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_syst_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_err_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_syst_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
sum_data_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_data_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_data_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_bkgd_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_syst_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_bkgd_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_syst_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_bkgd_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_syst_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_significance_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_significance_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_significance_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_excess_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_excess_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_excess_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_syst_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_allE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_allE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_syst_sky_map_allE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_syst_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_syst_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_LE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_LE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_syst_sky_map_LE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_HE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_HE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_syst_sky_map_HE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)

sum_mimic_incl_sky_map = []
sum_mimic_data_sky_map = []
sum_mimic_bkgd_sky_map = []
sum_mimic_incl_sky_map_smooth = []
sum_mimic_data_sky_map_smooth = []
sum_mimic_bkgd_sky_map_smooth = []
sum_mimic_excess_sky_map_smooth = []
sum_mimic_significance_sky_map = []
sum_mimic_flux_sky_map = []
sum_mimic_flux_err_sky_map = []
sum_mimic_flux_sky_map_smooth = []
sum_mimic_flux_err_sky_map_smooth = []
sum_mimic_data_sky_map_allE = []
sum_mimic_data_sky_map_LE = []
sum_mimic_data_sky_map_HE = []
sum_mimic_bkgd_sky_map_allE = []
sum_mimic_bkgd_sky_map_LE = []
sum_mimic_bkgd_sky_map_HE = []
sum_mimic_significance_sky_map_allE = []
sum_mimic_significance_sky_map_LE = []
sum_mimic_significance_sky_map_HE = []
sum_mimic_excess_sky_map_allE = []
sum_mimic_excess_sky_map_LE = []
sum_mimic_excess_sky_map_HE = []
sum_mimic_flux_sky_map_allE = []
sum_mimic_flux_err_sky_map_allE = []
sum_mimic_flux_sky_map_allE_smooth = []
sum_mimic_flux_err_sky_map_allE_smooth = []
sum_mimic_flux_sky_map_LE = []
sum_mimic_flux_err_sky_map_LE = []
sum_mimic_flux_sky_map_HE = []
sum_mimic_flux_err_sky_map_HE = []
sum_mimic_flux_sky_map_LE_smooth = []
sum_mimic_flux_err_sky_map_LE_smooth = []
sum_mimic_flux_sky_map_HE_smooth = []
sum_mimic_flux_err_sky_map_HE_smooth = []
for mimic in range(1,n_mimic+1):
    mimic_incl_sky_map = []
    mimic_data_sky_map = []
    mimic_bkgd_sky_map = []
    mimic_incl_sky_map_smooth = []
    mimic_data_sky_map_smooth = []
    mimic_bkgd_sky_map_smooth = []
    mimic_excess_sky_map_smooth = []
    mimic_significance_sky_map = []
    mimic_flux_sky_map = []
    mimic_flux_err_sky_map = []
    mimic_flux_sky_map_smooth = []
    mimic_flux_err_sky_map_smooth = []
    for logE in range(0,logE_nbins):
        mimic_incl_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_incl_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_data_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_bkgd_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_excess_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_significance_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_flux_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_flux_err_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_flux_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        mimic_flux_err_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_mimic_incl_sky_map += [mimic_incl_sky_map]
    sum_mimic_data_sky_map += [mimic_data_sky_map]
    sum_mimic_bkgd_sky_map += [mimic_bkgd_sky_map]
    sum_mimic_incl_sky_map_smooth += [mimic_incl_sky_map_smooth]
    sum_mimic_data_sky_map_smooth += [mimic_data_sky_map_smooth]
    sum_mimic_bkgd_sky_map_smooth += [mimic_bkgd_sky_map_smooth]
    sum_mimic_excess_sky_map_smooth += [mimic_excess_sky_map_smooth]
    sum_mimic_significance_sky_map += [mimic_significance_sky_map]
    sum_mimic_flux_sky_map += [mimic_flux_sky_map]
    sum_mimic_flux_err_sky_map += [mimic_flux_err_sky_map]
    sum_mimic_flux_sky_map_smooth += [mimic_flux_sky_map_smooth]
    sum_mimic_flux_err_sky_map_smooth += [mimic_flux_err_sky_map_smooth]
for mimic in range(1,n_mimic+1):
    mimic_data_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_data_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_data_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_bkgd_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_bkgd_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_bkgd_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_significance_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_significance_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_significance_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_excess_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_excess_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_excess_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_err_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_sky_map_allE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_err_sky_map_allE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_err_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_err_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_sky_map_LE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_err_sky_map_LE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_sky_map_HE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    mimic_flux_err_sky_map_HE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    sum_mimic_data_sky_map_allE += [mimic_data_sky_map_allE]
    sum_mimic_data_sky_map_LE += [mimic_data_sky_map_LE]
    sum_mimic_data_sky_map_HE += [mimic_data_sky_map_HE]
    sum_mimic_bkgd_sky_map_allE += [mimic_bkgd_sky_map_allE]
    sum_mimic_bkgd_sky_map_LE += [mimic_bkgd_sky_map_LE]
    sum_mimic_bkgd_sky_map_HE += [mimic_bkgd_sky_map_HE]
    sum_mimic_significance_sky_map_allE += [mimic_significance_sky_map_allE]
    sum_mimic_significance_sky_map_LE += [mimic_significance_sky_map_LE]
    sum_mimic_significance_sky_map_HE += [mimic_significance_sky_map_HE]
    sum_mimic_excess_sky_map_allE += [mimic_excess_sky_map_allE]
    sum_mimic_excess_sky_map_LE += [mimic_excess_sky_map_LE]
    sum_mimic_excess_sky_map_HE += [mimic_excess_sky_map_HE]
    sum_mimic_flux_sky_map_allE += [mimic_flux_sky_map_allE]
    sum_mimic_flux_err_sky_map_allE += [mimic_flux_err_sky_map_allE]
    sum_mimic_flux_sky_map_allE_smooth += [mimic_flux_sky_map_allE_smooth]
    sum_mimic_flux_err_sky_map_allE_smooth += [mimic_flux_err_sky_map_allE_smooth]
    sum_mimic_flux_sky_map_LE += [mimic_flux_sky_map_LE]
    sum_mimic_flux_err_sky_map_LE += [mimic_flux_err_sky_map_LE]
    sum_mimic_flux_sky_map_HE += [mimic_flux_sky_map_HE]
    sum_mimic_flux_err_sky_map_HE += [mimic_flux_err_sky_map_HE]
    sum_mimic_flux_sky_map_LE_smooth += [mimic_flux_sky_map_LE_smooth]
    sum_mimic_flux_err_sky_map_LE_smooth += [mimic_flux_err_sky_map_LE_smooth]
    sum_mimic_flux_sky_map_HE_smooth += [mimic_flux_sky_map_HE_smooth]
    sum_mimic_flux_err_sky_map_HE_smooth += [mimic_flux_err_sky_map_HE_smooth]


sum_data_xyoff_map = []
sum_fit_xyoff_map = []
sum_ratio_xyoff_map = []
sum_err_xyoff_map = []
sum_init_err_xyoff_map = []
for logE in range(0,logE_nbins):
    sum_data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_ratio_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_err_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_init_err_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

n_groups = 0.
for epoch in input_epoch:

    onoff_list = []
    onoff_list += [onoff]
    if onoff=='ON':
        for mimic in range(1,n_mimic+1):
            onoff_list += [f'MIMIC{mimic}']

    for mode in onoff_list:

        input_filename = f'{smi_output}/skymaps_{source_name}_{epoch}_{mode}_{ana_tag}.pkl'
        print (f'reading {input_filename}...')
        if not os.path.exists(input_filename):
            print (f'{input_filename} does not exist.')
            continue
        analysis_result = pickle.load(open(input_filename, "rb"))
        
        for run in range(0,len(analysis_result)):

            run_info = analysis_result[run][0] 
            exposure = run_info[0]
            run_elev = run_info[1]
            run_azim = run_info[2]
            truth_params = run_info[3]
            fit_params = run_info[4]
            run_nsb = run_info[5]

            if run_azim>270.:
                run_azim = run_azim-360.

            if run_elev<elev_cut:
                continue

            if not 'MIMIC' in mode:
                total_exposure += exposure
            else:
                mimic_index = int(mode.strip('MIMIC'))
                mimic_exposure[mimic_index-1] += exposure

            is_good_run = True
            if not is_good_run: 
                print (f'bad fitting. reject the run.')
                continue

            incl_sky_map = analysis_result[run][1] 
            data_sky_map = analysis_result[run][2] 
            bkgd_sky_map = analysis_result[run][3] 
            syst_sky_map = analysis_result[run][4] 
            data_xyoff_map = analysis_result[run][5]
            fit_xyoff_map = analysis_result[run][6]
            ratio_xyoff_map = analysis_result[run][7]

            if not 'MIMIC' in mode:
                good_exposure += exposure
                list_run_elev += [run_elev]
                list_run_azim += [run_azim]
                list_truth_params += [truth_params]
                list_fit_params += [fit_params]
                n_groups += 1.

            logE_peak = 0
            bkgd_peak = 0.
            for logE in range(0,logE_nbins):
                bkgd = np.sum(fit_xyoff_map[logE].waxis[:,:,:])
                if bkgd>bkgd_peak:
                    bkgd_peak = bkgd
                    logE_peak = logE

            for logE in range(0,logE_nbins):
                if logE<logE_peak: continue
                if logE<logE_min: continue
                if logE>logE_max: continue

                if 'MIMIC' in mode:
                    mimic_index = int(mode.strip('MIMIC'))-1
                    sum_mimic_incl_sky_map[mimic_index][logE].add(incl_sky_map[logE])
                    sum_mimic_data_sky_map[mimic_index][logE].add(data_sky_map[logE])
                    sum_mimic_bkgd_sky_map[mimic_index][logE].add(bkgd_sky_map[logE])
                else:
                    sum_incl_sky_map[logE].add(incl_sky_map[logE])
                    sum_data_sky_map[logE].add(data_sky_map[logE])
                    sum_bkgd_sky_map[logE].add(bkgd_sky_map[logE])
                    sum_syst_sky_map[logE].add(syst_sky_map[logE])
                    sum_data_xyoff_map[logE].add(data_xyoff_map[logE])
                    sum_fit_xyoff_map[logE].add(fit_xyoff_map[logE])
                    sum_ratio_xyoff_map[logE].add(ratio_xyoff_map[logE])
    
for logE in range(0,logE_nbins):
    data_integral = 0.
    model_integral = 0.
    for idx_x in range(0,xoff_bins[logE]):
        for idx_y in range(0,yoff_bins[logE]):
            data_integral += sum_data_xyoff_map[logE].waxis[idx_x,idx_y,0] 
            model_integral += sum_data_xyoff_map[logE].waxis[idx_x,idx_y,1] 
    if model_integral==0.: continue
    for idx_x in range(0,xoff_bins[logE]):
        for idx_y in range(0,yoff_bins[logE]):
            data = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,0] 
            model = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,1]*data_integral/model_integral 
            data_err = max(1.,pow(data,0.5))
            sum_init_err_xyoff_map[logE].waxis[idx_x,idx_y,0] = (data-model)/data_err

for logE in range(0,logE_nbins):
    for idx_x in range(0,xoff_bins[logE]):
        for idx_y in range(0,yoff_bins[logE]):
            for gcut in range(0,gcut_bins):
                data = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                model = sum_fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                significance = significance_li_and_ma(data, model, 0.)
                sum_err_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = significance

for logE in range(0,logE_nbins):
    sum_data_xyoff_map[logE].scale(1./good_exposure)
    sum_fit_xyoff_map[logE].scale(1./good_exposure)
    sum_ratio_xyoff_map[logE].scale(1./n_groups)

for logE in range(0,logE_nbins):
    sum_incl_sky_map_smooth[logE].reset()
    sum_data_sky_map_smooth[logE].reset()
    sum_bkgd_sky_map_smooth[logE].reset()
    sum_excess_sky_map_smooth[logE].reset()
    sum_incl_sky_map_smooth[logE].add(sum_incl_sky_map[logE])
    sum_data_sky_map_smooth[logE].add(sum_data_sky_map[logE])
    sum_bkgd_sky_map_smooth[logE].add(sum_bkgd_sky_map[logE])
    sum_syst_sky_map_smooth[logE].add(sum_syst_sky_map[logE])
    smooth_image(sum_incl_sky_map_smooth[logE].waxis[:,:,0],sum_incl_sky_map_smooth[logE].xaxis,sum_incl_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
    smooth_image(sum_bkgd_sky_map_smooth[logE].waxis[:,:,0],sum_bkgd_sky_map_smooth[logE].xaxis,sum_bkgd_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
    smooth_image(sum_syst_sky_map_smooth[logE].waxis[:,:,0],sum_syst_sky_map_smooth[logE].xaxis,sum_syst_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
    smooth_image(sum_data_sky_map_smooth[logE].waxis[:,:,0],sum_data_sky_map_smooth[logE].xaxis,sum_data_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
    sum_data_sky_map_allE.add(sum_data_sky_map_smooth[logE])
    sum_bkgd_sky_map_allE.add(sum_bkgd_sky_map_smooth[logE])
    sum_syst_sky_map_allE.add(sum_syst_sky_map_smooth[logE])
    if logE>=logE_min and logE<logE_mid:
        sum_data_sky_map_LE.add(sum_data_sky_map_smooth[logE])
        sum_bkgd_sky_map_LE.add(sum_bkgd_sky_map_smooth[logE])
        sum_syst_sky_map_LE.add(sum_syst_sky_map_smooth[logE])
    if logE>=logE_mid and logE<=logE_max:
        sum_data_sky_map_HE.add(sum_data_sky_map_smooth[logE])
        sum_bkgd_sky_map_HE.add(sum_bkgd_sky_map_smooth[logE])
        sum_syst_sky_map_HE.add(sum_syst_sky_map_smooth[logE])

for logE in range(0,logE_nbins):
    for mimic in range(0,n_mimic):
        sum_mimic_incl_sky_map_smooth[mimic][logE].reset()
        sum_mimic_data_sky_map_smooth[mimic][logE].reset()
        sum_mimic_bkgd_sky_map_smooth[mimic][logE].reset()
        sum_mimic_excess_sky_map_smooth[mimic][logE].reset()
        sum_mimic_incl_sky_map_smooth[mimic][logE].add(sum_mimic_incl_sky_map[mimic][logE])
        sum_mimic_data_sky_map_smooth[mimic][logE].add(sum_mimic_data_sky_map[mimic][logE])
        sum_mimic_bkgd_sky_map_smooth[mimic][logE].add(sum_mimic_bkgd_sky_map[mimic][logE])
        smooth_image(sum_mimic_incl_sky_map_smooth[mimic][logE].waxis[:,:,0],sum_incl_sky_map_smooth[logE].xaxis,sum_incl_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
        smooth_image(sum_mimic_bkgd_sky_map_smooth[mimic][logE].waxis[:,:,0],sum_bkgd_sky_map_smooth[logE].xaxis,sum_bkgd_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
        smooth_image(sum_mimic_data_sky_map_smooth[mimic][logE].waxis[:,:,0],sum_data_sky_map_smooth[logE].xaxis,sum_data_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
        sum_mimic_data_sky_map_allE[mimic].add(sum_mimic_data_sky_map_smooth[mimic][logE])
        sum_mimic_bkgd_sky_map_allE[mimic].add(sum_mimic_bkgd_sky_map_smooth[mimic][logE])
        if logE>=logE_min and logE<logE_mid:
            sum_mimic_data_sky_map_LE[mimic].add(sum_mimic_data_sky_map_smooth[mimic][logE])
            sum_mimic_bkgd_sky_map_LE[mimic].add(sum_mimic_bkgd_sky_map_smooth[mimic][logE])
        if logE>=logE_mid and logE<=logE_max:
            sum_mimic_data_sky_map_HE[mimic].add(sum_mimic_data_sky_map_smooth[mimic][logE])
            sum_mimic_bkgd_sky_map_HE[mimic].add(sum_mimic_bkgd_sky_map_smooth[mimic][logE])


print ('=================================================================================')
for logE in range(0,logE_nbins):

    data_sum = np.sum(sum_data_sky_map[logE].waxis[:,:,0])
    bkgd_sum = np.sum(sum_bkgd_sky_map[logE].waxis[:,:,0])

    error = 0.
    stat_error = 0.
    if data_sum>0.:
        error = 100.*(data_sum-bkgd_sum)/data_sum
        stat_error = 100.*pow(data_sum,0.5)/data_sum
    print (f'E = {pow(10.,logE_bins[logE]):0.3f} TeV, data_sum = {data_sum}, bkgd_sum = {bkgd_sum:0.1f}, error = {error:0.1f} +/- {stat_error:0.1f} %')

for logE in range(0,logE_nbins):
    make_significance_map(sum_data_sky_map_smooth[logE],sum_bkgd_sky_map_smooth[logE],sum_significance_sky_map[logE],sum_excess_sky_map_smooth[logE],syst_sky_map=sum_syst_sky_map_smooth[logE])
make_significance_map(sum_data_sky_map_allE,sum_bkgd_sky_map_allE,sum_significance_sky_map_allE,sum_excess_sky_map_allE,syst_sky_map=sum_syst_sky_map_allE)
make_significance_map(sum_data_sky_map_LE,sum_bkgd_sky_map_LE,sum_significance_sky_map_LE,sum_excess_sky_map_LE,syst_sky_map=sum_syst_sky_map_LE)
make_significance_map(sum_data_sky_map_HE,sum_bkgd_sky_map_HE,sum_significance_sky_map_HE,sum_excess_sky_map_HE,syst_sky_map=sum_syst_sky_map_HE)

for mimic in range(0,n_mimic):
    make_significance_map(sum_mimic_data_sky_map_allE[mimic],sum_mimic_bkgd_sky_map_allE[mimic],sum_mimic_significance_sky_map_allE[mimic],sum_mimic_excess_sky_map_allE[mimic])

for logE in range(0,logE_nbins):
    avg_energy = 0.5*(pow(10.,logE_axis.xaxis[logE])+pow(10.,logE_axis.xaxis[logE+1]))
    delta_energy = 0.5*(pow(10.,logE_axis.xaxis[logE+1])-pow(10.,logE_axis.xaxis[logE]))
    make_flux_map(sum_incl_sky_map_smooth[logE],sum_data_sky_map_smooth[logE],sum_bkgd_sky_map_smooth[logE],sum_flux_sky_map_smooth[logE],sum_flux_err_sky_map_smooth[logE],sum_flux_syst_sky_map_smooth[logE],avg_energy,delta_energy,syst_sky_map=sum_syst_sky_map_smooth[logE])
    make_flux_map(sum_incl_sky_map_smooth[logE],sum_data_sky_map[logE],sum_bkgd_sky_map[logE],sum_flux_sky_map[logE],sum_flux_err_sky_map[logE],sum_flux_syst_sky_map[logE],avg_energy,delta_energy,syst_sky_map=sum_syst_sky_map[logE])

for mimic in range(0,n_mimic):
    for logE in range(0,logE_nbins):
        make_flux_map(sum_mimic_incl_sky_map_smooth[mimic][logE],sum_mimic_data_sky_map[mimic][logE],sum_mimic_bkgd_sky_map[mimic][logE],sum_mimic_flux_sky_map[mimic][logE],sum_mimic_flux_err_sky_map[mimic][logE],avg_energy,delta_energy)

for logE in range(logE_min,logE_max):
    PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE,logE+1,sum_flux_sky_map_smooth[logE],f'{source_name}_flux_sky_map_logE{logE}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)
    PlotSkyMap(fig,'background count',logE,logE+1,sum_bkgd_sky_map_smooth[logE],f'{source_name}_bkgd_sky_map_logE{logE}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',layer=0,zoomin=zoomin)
    PlotSkyMap(fig,'excess count',logE,logE+1,sum_excess_sky_map_smooth[logE],f'{source_name}_excess_sky_map_logE{logE}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',layer=0,zoomin=zoomin)
    sum_flux_sky_map_allE.add(sum_flux_sky_map[logE])
    sum_flux_err_sky_map_allE.addSquare(sum_flux_err_sky_map[logE])
    sum_flux_syst_sky_map_allE.add(sum_flux_syst_sky_map[logE])
    sum_flux_sky_map_allE_smooth.add(sum_flux_sky_map_smooth[logE])
    sum_flux_err_sky_map_allE_smooth.addSquare(sum_flux_err_sky_map_smooth[logE])
    sum_flux_syst_sky_map_allE_smooth.add(sum_flux_syst_sky_map_smooth[logE])
    for mimic in range(0,n_mimic):
        sum_mimic_flux_sky_map_allE[mimic].add(sum_mimic_flux_sky_map[mimic][logE])
        sum_mimic_flux_err_sky_map_allE[mimic].addSquare(sum_mimic_flux_err_sky_map[mimic][logE])
    if logE<logE_min: continue
    if logE>logE_max: continue
    if logE>=logE_min and logE<logE_mid:
        sum_flux_sky_map_LE.add(sum_flux_sky_map[logE])
        sum_flux_err_sky_map_LE.addSquare(sum_flux_err_sky_map[logE])
        sum_flux_syst_sky_map_LE.add(sum_flux_syst_sky_map[logE])
        sum_flux_sky_map_LE_smooth.add(sum_flux_sky_map_smooth[logE])
        sum_flux_err_sky_map_LE_smooth.addSquare(sum_flux_err_sky_map_smooth[logE])
        sum_flux_syst_sky_map_LE_smooth.add(sum_flux_syst_sky_map_smooth[logE])
        for mimic in range(0,n_mimic):
            sum_mimic_flux_sky_map_LE[mimic].add(sum_mimic_flux_sky_map[mimic][logE])
            sum_mimic_flux_err_sky_map_LE[mimic].addSquare(sum_mimic_flux_err_sky_map[mimic][logE])
    if logE>=logE_mid and logE<=logE_max:
        sum_flux_sky_map_HE.add(sum_flux_sky_map[logE])
        sum_flux_err_sky_map_HE.addSquare(sum_flux_err_sky_map[logE])
        sum_flux_syst_sky_map_HE.add(sum_flux_syst_sky_map[logE])
        sum_flux_sky_map_HE_smooth.add(sum_flux_sky_map_smooth[logE])
        sum_flux_err_sky_map_HE_smooth.addSquare(sum_flux_err_sky_map_smooth[logE])
        sum_flux_syst_sky_map_HE_smooth.add(sum_flux_syst_sky_map_smooth[logE])
        for mimic in range(0,n_mimic):
            sum_mimic_flux_sky_map_HE[mimic].add(sum_mimic_flux_sky_map[mimic][logE])
            sum_mimic_flux_err_sky_map_HE[mimic].addSquare(sum_mimic_flux_err_sky_map[mimic][logE])
PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE_min,logE_max,sum_flux_sky_map_allE_smooth,f'{source_name}_flux_sky_map_allE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)
PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE_min,logE_mid,sum_flux_sky_map_LE_smooth,f'{source_name}_flux_sky_map_LE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)
PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE_mid,logE_max,sum_flux_sky_map_HE_smooth,f'{source_name}_flux_sky_map_HE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('Fit 2d Gaussian (LE)')
fit_2d_model(sum_data_sky_map_LE, sum_bkgd_sky_map_LE, all_roi_x[0], all_roi_y[0])
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('Fit 2d Gaussian (HE)')
fit_2d_model(sum_data_sky_map_HE, sum_bkgd_sky_map_HE, all_roi_x[0], all_roi_y[0])
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('Fit 2d Gaussian (sum)')
fit_2d_model(sum_data_sky_map_allE, sum_bkgd_sky_map_allE, all_roi_x[0], all_roi_y[0])
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

low_energy = int(1000.*pow(10.,logE_bins[logE_min]))
high_energy = int(1000.*pow(10.,logE_bins[logE_mid]))
SaveFITS(sum_flux_sky_map_LE_smooth,f'sum_flux_sky_map_{low_energy}GeV_{high_energy}GeV')
SaveFITS(sum_flux_err_sky_map_LE_smooth,f'sum_flux_err_sky_map_{low_energy}GeV_{high_energy}GeV')
low_energy = int(1000.*pow(10.,logE_bins[logE_mid]))
high_energy = int(1000.*pow(10.,logE_bins[logE_max]))
SaveFITS(sum_flux_sky_map_HE_smooth,f'sum_flux_sky_map_{low_energy}GeV_{high_energy}GeV')
SaveFITS(sum_flux_err_sky_map_HE_smooth,f'sum_flux_err_sky_map_{low_energy}GeV_{high_energy}GeV')
for logE in range(logE_min,logE_max):
    low_energy = int(1000.*pow(10.,logE_bins[logE]))
    high_energy = int(1000.*pow(10.,logE_bins[logE+1]))
    SaveFITS(sum_flux_sky_map_smooth[logE],f'sum_flux_sky_map_{low_energy}GeV_{high_energy}GeV')
    SaveFITS(sum_flux_err_sky_map_smooth[logE],f'sum_flux_err_sky_map_{low_energy}GeV_{high_energy}GeV')

PrintAndPlotInformationRoI(fig,logE_min,logE_mid,logE_max,source_name,sum_data_sky_map,sum_bkgd_sky_map,sum_syst_sky_map,sum_flux_sky_map,sum_flux_err_sky_map,sum_flux_syst_sky_map,sum_mimic_data_sky_map,sum_mimic_bkgd_sky_map,all_roi_name,all_roi_x,all_roi_y,all_roi_r,all_excl_x,all_excl_y,all_excl_r)

    
for logE in range(logE_min,logE_max):
    on_radial_axis, on_profile_axis, on_profile_err_axis, on_profile_syst_axis = GetRadialProfile(sum_flux_sky_map[logE],sum_flux_err_sky_map[logE],sum_flux_syst_sky_map[logE],all_roi_x,all_roi_y,2.0,all_excl_x,all_excl_y,all_excl_r)
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
    axbig.fill_between(on_radial_axis,np.array(baseline_yaxis)-np.array(on_profile_syst_axis),np.array(baseline_yaxis)+np.array(on_profile_syst_axis),alpha=0.2,color='b')
    axbig.plot(on_radial_axis, baseline_yaxis, color='b', ls='dashed')
    axbig.errorbar(on_radial_axis,on_profile_axis,on_profile_err_axis,color='k',marker='+',ls='none')
    fig.savefig(f'output_plots/{source_name}_surface_brightness_logE{logE}_{all_roi_name[0]}_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

on_radial_axis, on_profile_axis, on_profile_err_axis, on_profile_syst_axis = GetRadialProfile(sum_flux_sky_map_allE,sum_flux_err_sky_map_allE,sum_flux_syst_sky_map_allE,all_roi_x,all_roi_y,2.0,all_excl_x,all_excl_y,all_excl_r)
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
axbig.fill_between(on_radial_axis,np.array(baseline_yaxis)-np.array(on_profile_syst_axis),np.array(baseline_yaxis)+np.array(on_profile_syst_axis),alpha=0.2,color='b')
axbig.plot(on_radial_axis, baseline_yaxis, color='b', ls='dashed')
axbig.errorbar(on_radial_axis,on_profile_axis,on_profile_err_axis,color='k',marker='+',ls='none')
fig.savefig(f'output_plots/{source_name}_surface_brightness_allE_{all_roi_name[0]}_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

on_radial_axis, on_profile_axis, on_profile_err_axis, on_profile_syst_axis = GetRadialProfile(sum_flux_sky_map_LE,sum_flux_err_sky_map_LE,sum_flux_syst_sky_map_LE,all_roi_x,all_roi_y,2.0,all_excl_x,all_excl_y,all_excl_r)
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
axbig.fill_between(on_radial_axis,np.array(baseline_yaxis)-np.array(on_profile_syst_axis),np.array(baseline_yaxis)+np.array(on_profile_syst_axis),alpha=0.2,color='b')
axbig.plot(on_radial_axis, baseline_yaxis, color='b', ls='dashed')
axbig.errorbar(on_radial_axis,on_profile_axis,on_profile_err_axis,color='k',marker='+',ls='none')
fig.savefig(f'output_plots/{source_name}_surface_brightness_LE_{all_roi_name[0]}_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

on_radial_axis, on_profile_axis, on_profile_err_axis, on_profile_syst_axis = GetRadialProfile(sum_flux_sky_map_HE,sum_flux_err_sky_map_HE,sum_flux_syst_sky_map_HE,all_roi_x,all_roi_y,2.0,all_excl_x,all_excl_y,all_excl_r)
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
axbig.fill_between(on_radial_axis,np.array(baseline_yaxis)-np.array(on_profile_syst_axis),np.array(baseline_yaxis)+np.array(on_profile_syst_axis),alpha=0.2,color='b')
axbig.plot(on_radial_axis, baseline_yaxis, color='b', ls='dashed')
axbig.errorbar(on_radial_axis,on_profile_axis,on_profile_err_axis,color='k',marker='+',ls='none')
fig.savefig(f'output_plots/{source_name}_surface_brightness_HE_{all_roi_name[0]}_{ana_tag}.png',bbox_inches='tight')
axbig.remove()


if 'PSR_J1856_p0245' in source_name:

    on_radial_axis = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
    on_profile_axis = [1.42630770e-09, 8.45610102e-10, 3.97927911e-10, 1.77387588e-10, 2.17602561e-10, 1.69687255e-10, 2.00294064e-11, -7.74294257e-11, 2.45221243e-11, -1.22693293e-11, -1.28535773e-11] #TeV/cm2/s/deg2
    on_profile_err_axis =[3.49403638e-10, 1.84587749e-10, 1.16395624e-10, 1.74538576e-10, 1.34151761e-10, 7.07628523e-11, 9.00952643e-11, 5.02971675e-11, 2.47404062e-11, 1.01558004e-11, 1.85267251e-13] #TeV/cm2/s/deg2
    profile_sum = np.sum(on_profile_axis)
    start = (profile_sum, 0.5)
    popt, pcov = curve_fit(diffusion_func,on_radial_axis,on_profile_axis,p0=start,sigma=on_profile_err_axis,absolute_sigma=True,bounds=((0, 0.01), (np.inf, np.inf)))
    profile_fit = diffusion_func(np.array(on_radial_axis), *popt)
    residual = np.array(on_profile_axis) - profile_fit
    chisq = np.sum((residual/np.array(on_profile_err_axis))**2)
    dof = len(on_radial_axis)-2
    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (f'Fermi data')
    print ('diffusion flux = %0.2E +/- %0.2E'%(popt[0],pow(pcov[0][0],0.5)))
    print ('diffusion radius = %0.2f +/- %0.2f deg (chi2/dof = %0.2f)'%(popt[1],pow(pcov[1][1],0.5),chisq/dof))

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
    axbig.errorbar(on_radial_axis,on_profile_axis,on_profile_err_axis,color='k',marker='+',ls='none',zorder=2)
    if fit_radial_profile:
        axbig.plot(on_radial_axis,diffusion_func(np.array(on_radial_axis),*popt),color='r')
    fig.savefig(f'output_plots/{source_name}_surface_brightness_Fermi_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

for logE in range(logE_min,logE_max):

    PlotSkyMap(fig,'significance',logE,logE+1,sum_significance_sky_map[logE],f'{source_name}_significance_sky_map_logE{logE}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=3.,zoomin=zoomin)

    max_z = 5.

    #fig.clf()
    #figsize_x = 7
    #figsize_y = 7
    #fig.set_figheight(figsize_y)
    #fig.set_figwidth(figsize_x)
    #axbig = fig.add_subplot()
    #label_x = 'Camera X'
    #label_y = 'Camera Y'
    #axbig.set_xlabel(label_x)
    #axbig.set_ylabel(label_y)
    #xmin = sum_init_err_xyoff_map[logE].xaxis.min()
    #xmax = sum_init_err_xyoff_map[logE].xaxis.max()
    #ymin = sum_init_err_xyoff_map[logE].yaxis.min()
    #ymax = sum_init_err_xyoff_map[logE].yaxis.max()
    #im = axbig.imshow(sum_init_err_xyoff_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
    #cbar = fig.colorbar(im)
    #fig.savefig(f'output_plots/{source_name}_xyoff_init_err_map_logE{logE}_{ana_tag}.png',bbox_inches='tight')
    #axbig.remove()

    #fig.clf()
    #figsize_x = 7
    #figsize_y = 7
    #fig.set_figheight(figsize_y)
    #fig.set_figwidth(figsize_x)
    #axbig = fig.add_subplot()
    #label_x = 'Camera X'
    #label_y = 'Camera Y'
    #axbig.set_xlabel(label_x)
    #axbig.set_ylabel(label_y)
    #xmin = sum_ratio_xyoff_map[logE].xaxis.min()
    #xmax = sum_ratio_xyoff_map[logE].xaxis.max()
    #ymin = sum_ratio_xyoff_map[logE].yaxis.min()
    #ymax = sum_ratio_xyoff_map[logE].yaxis.max()
    #im = axbig.imshow(sum_ratio_xyoff_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    #cbar = fig.colorbar(im)
    #fig.savefig(f'output_plots/{source_name}_xyoff_map_logE{logE}_ratio_{ana_tag}.png',bbox_inches='tight')
    #axbig.remove()

    #fig.clf()
    #figsize_x = 7
    #figsize_y = 7
    #fig.set_figheight(figsize_y)
    #fig.set_figwidth(figsize_x)
    #axbig = fig.add_subplot()
    #label_x = 'Camera X'
    #label_y = 'Camera Y'
    #axbig.set_xlabel(label_x)
    #axbig.set_ylabel(label_y)
    #xmin = sum_data_xyoff_map[logE].xaxis.min()
    #xmax = sum_data_xyoff_map[logE].xaxis.max()
    #ymin = sum_data_xyoff_map[logE].yaxis.min()
    #ymax = sum_data_xyoff_map[logE].yaxis.max()
    #im = axbig.imshow(sum_data_xyoff_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    #cbar = fig.colorbar(im)
    #fig.savefig(f'output_plots/{source_name}_xyoff_map_logE{logE}_data_{ana_tag}.png',bbox_inches='tight')
    #axbig.remove()

    fig.clf()
    figsize_x = 7
    figsize_y = 7
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'Camera X'
    label_y = 'Camera Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = sum_data_xyoff_map[logE].xaxis.min()
    xmax = sum_data_xyoff_map[logE].xaxis.max()
    ymin = sum_data_xyoff_map[logE].yaxis.min()
    ymax = sum_data_xyoff_map[logE].yaxis.max()
    im = axbig.imshow(sum_data_xyoff_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    cbar = fig.colorbar(im)
    fig.savefig(f'output_plots/{source_name}_xyoff_data_map_logE{logE}_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    figsize_x = 7
    figsize_y = 7
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'Camera X'
    label_y = 'Camera Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = sum_data_xyoff_map[logE].xaxis.min()
    xmax = sum_data_xyoff_map[logE].xaxis.max()
    ymin = sum_data_xyoff_map[logE].yaxis.min()
    ymax = sum_data_xyoff_map[logE].yaxis.max()
    im = axbig.imshow(sum_fit_xyoff_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    cbar = fig.colorbar(im)
    fig.savefig(f'output_plots/{source_name}_xyoff_fit_map_logE{logE}_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    figsize_x = 7
    figsize_y = 7
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'Camera X'
    label_y = 'Camera Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = sum_err_xyoff_map[logE].xaxis.min()
    xmax = sum_err_xyoff_map[logE].xaxis.max()
    ymin = sum_err_xyoff_map[logE].yaxis.min()
    ymax = sum_err_xyoff_map[logE].yaxis.max()
    im = axbig.imshow(sum_err_xyoff_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
    cbar = fig.colorbar(im)
    fig.savefig(f'output_plots/{source_name}_xyoff_err_map_logE{logE}_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

fig.clf()
figsize_y = 2.*gcut_bins
figsize_x = 2.*(logE_max-logE_min)
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
ax_idx = 0
gs = GridSpec(gcut_bins, (logE_max-logE_min), hspace=0.1, wspace=0.1)
for logE in range(logE_min,logE_max):
    for gcut in range(0,gcut_bins):
        ax_idx = logE-logE_min + (logE_max-logE_min)*gcut + 1
        axbig = fig.add_subplot(gs[ax_idx-1])
        if logE==logE_min:
            if gcut==0:
                axbig.set_ylabel('SR')
            else:
                axbig.set_ylabel(f'CR{gcut}')
        if gcut==0:
            axbig.set_title(f'{pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
        if not logE==logE_min:
            axbig.axes.get_yaxis().set_visible(False)
        if not gcut==gcut_bins-1:
            axbig.axes.get_xaxis().set_visible(False)
        xmin = sum_data_xyoff_map[logE].xaxis.min()
        xmax = sum_data_xyoff_map[logE].xaxis.max()
        ymin = sum_data_xyoff_map[logE].yaxis.min()
        ymax = sum_data_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(sum_data_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
fig.savefig(f'output_plots/{source_name}_xyoff_map_inclusive_data_transpose_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

fig.clf()
figsize_x = 2.*gcut_bins
figsize_y = 2.*(logE_max-logE_min)
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
ax_idx = 0
for logE in range(logE_min,logE_max):
    for gcut in range(0,gcut_bins):
        ax_idx = gcut + (logE-logE_min)*gcut_bins + 1
        axbig = fig.add_subplot((logE_max-logE_min),gcut_bins,ax_idx)
        if logE==logE_min:
            if gcut==0:
                axbig.set_title('SR')
            else:
                axbig.set_title(f'CR{gcut}')
        if gcut==0:
            axbig.set_ylabel(f'E = {pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
        xmin = sum_data_xyoff_map[logE].xaxis.min()
        xmax = sum_data_xyoff_map[logE].xaxis.max()
        ymin = sum_data_xyoff_map[logE].yaxis.min()
        ymax = sum_data_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(sum_data_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
fig.savefig(f'output_plots/{source_name}_xyoff_map_inclusive_data_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

fig.clf()
figsize_y = 2.*gcut_bins
figsize_x = 2.*(logE_max-logE_min)
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
ax_idx = 0
gs = GridSpec(gcut_bins, (logE_max-logE_min), hspace=0.1, wspace=0.1)
for logE in range(logE_min,logE_max):
    for gcut in range(0,gcut_bins):
        ax_idx = logE-logE_min + (logE_max-logE_min)*gcut + 1
        axbig = fig.add_subplot(gs[ax_idx-1])
        if logE==logE_min:
            if gcut==0:
                axbig.set_ylabel('SR')
            else:
                axbig.set_ylabel(f'CR{gcut}')
        if gcut==0:
            axbig.set_title(f'{pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
        if not logE==logE_min:
            axbig.axes.get_yaxis().set_visible(False)
        if not gcut==gcut_bins-1:
            axbig.axes.get_xaxis().set_visible(False)
        xmin = sum_data_xyoff_map[logE].xaxis.min()
        xmax = sum_data_xyoff_map[logE].xaxis.max()
        ymin = sum_data_xyoff_map[logE].yaxis.min()
        ymax = sum_data_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(sum_err_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto',vmin=-max_z,vmax=max_z,cmap='coolwarm')
fig.savefig(f'output_plots/{source_name}_xyoff_map_inclusive_err_transpose_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

fig.clf()
figsize_x = 2.*gcut_bins
figsize_y = 2.*(logE_max-logE_min)
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
ax_idx = 0
for logE in range(logE_min,logE_max):
    for gcut in range(0,gcut_bins):
        ax_idx = gcut + (logE-logE_min)*gcut_bins + 1
        axbig = fig.add_subplot((logE_max-logE_min),gcut_bins,ax_idx)
        if logE==logE_min:
            if gcut==0:
                axbig.set_title('SR')
            else:
                axbig.set_title(f'CR{gcut}')
        if gcut==0:
            axbig.set_ylabel(f'E = {pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
        xmin = sum_data_xyoff_map[logE].xaxis.min()
        xmax = sum_data_xyoff_map[logE].xaxis.max()
        ymin = sum_data_xyoff_map[logE].yaxis.min()
        ymax = sum_data_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(sum_err_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto',vmin=-max_z,vmax=max_z,cmap='coolwarm')
fig.savefig(f'output_plots/{source_name}_xyoff_map_inclusive_err_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

#fig.clf()
#figsize_x = 3.*gcut_bins
#figsize_y = 24.
#fig.set_figheight(figsize_y)
#fig.set_figwidth(figsize_x)
#ax_idx = 0
#for logE in range(logE_min,logE_min+1):
#    for gcut in range(0,gcut_bins):
#        ax_idx = gcut + (logE-logE_min)*gcut_bins + 1
#        axbig = fig.add_subplot((logE_max-logE_min),gcut_bins,ax_idx)
#        if logE==logE_min:
#            if gcut==0:
#                axbig.set_title('SR')
#            else:
#                axbig.set_title(f'CR{gcut}')
#        if gcut==0:
#            axbig.set_ylabel(f'E = {pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
#        xmin = sum_data_xyoff_map[logE].xaxis.min()
#        xmax = sum_data_xyoff_map[logE].xaxis.max()
#        ymin = sum_data_xyoff_map[logE].yaxis.min()
#        ymax = sum_data_xyoff_map[logE].yaxis.max()
#        im = axbig.imshow(sum_err_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto',vmin=-max_z,vmax=max_z,cmap='coolwarm')
#fig.savefig(f'output_plots/{source_name}_xyoff_map_inclusive_err_{ana_tag}.png',bbox_inches='tight')
#axbig.remove()

PlotSkyMap(fig,'significance',logE_min,logE_max,sum_significance_sky_map_allE,f'{source_name}_significance_sky_map_allE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=3.,zoomin=zoomin)
PlotSkyMap(fig,'excess count',logE_min,logE_max,sum_excess_sky_map_allE,f'{source_name}_excess_sky_map_allE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)
PlotSkyMap(fig,'significance',logE_min,logE_mid,sum_significance_sky_map_LE,f'{source_name}_significance_sky_map_LE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=3.,zoomin=zoomin)
PlotSkyMap(fig,'excess count',logE_min,logE_mid,sum_excess_sky_map_LE,f'{source_name}_excess_sky_map_LE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)
PlotSkyMap(fig,'significance',logE_mid,logE_max,sum_significance_sky_map_HE,f'{source_name}_significance_sky_map_HE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=3.,zoomin=zoomin)
PlotSkyMap(fig,'excess count',logE_mid,logE_max,sum_excess_sky_map_HE,f'{source_name}_excess_sky_map_HE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')

for mimic in range(0,n_mimic):
    PlotSkyMap(fig,'significance',logE_min,logE_max,sum_mimic_significance_sky_map_allE[mimic],f'{source_name}_significance_sky_map_allE_mimic{mimic}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=3.,zoomin=zoomin)

print (f'total_exposure = {total_exposure}')
print (f'good_exposure = {good_exposure}')
print (f'mimic_exposure = {mimic_exposure}')

PlotCountProjection(fig,'count',logE_min,logE_max,sum_data_sky_map_allE,sum_bkgd_sky_map_allE,f'{source_name}_projection_sky_map_allE_{ana_tag}',hist_map_syst=sum_syst_sky_map_allE,roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
PlotCountProjection(fig,'count',logE_min,logE_mid,sum_data_sky_map_LE,sum_bkgd_sky_map_LE,f'{source_name}_projection_sky_map_LE_{ana_tag}',hist_map_syst=sum_syst_sky_map_LE,roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
PlotCountProjection(fig,'count',logE_mid,logE_max,sum_data_sky_map_HE,sum_bkgd_sky_map_HE,f'{source_name}_projection_sky_map_HE_{ana_tag}',hist_map_syst=sum_syst_sky_map_HE,roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')

for mimic in range(0,n_mimic):
    PlotCountProjection(fig,'count',logE_min,logE_max,sum_mimic_data_sky_map_allE[mimic],sum_mimic_bkgd_sky_map_allE[mimic],f'{source_name}_projection_sky_map_allE_mimic{mimic}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
    PlotCountProjection(fig,'count',logE_min,logE_mid,sum_mimic_data_sky_map_LE[mimic],sum_mimic_bkgd_sky_map_LE[mimic],f'{source_name}_projection_sky_map_LE_mimic{mimic}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
    PlotCountProjection(fig,'count',logE_mid,logE_max,sum_mimic_data_sky_map_HE[mimic],sum_mimic_bkgd_sky_map_HE[mimic],f'{source_name}_projection_sky_map_HE_mimic{mimic}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')


if 'PSR_J2021_p4026' in source_name:
    HI_sky_map = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    MWL_map_file = '/nevis/ged/data/rshang/MWL_maps/CGPS/CGPS_MO2_HI_line_image.fits' 
    GetSlicedDataCubeMap(MWL_map_file, HI_sky_map, -3., 5.)
    PlotSkyMap(fig,'Intensity',logE_min,logE_max,HI_sky_map,f'{source_name}_HI_sky_map_m03_p05_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)
    GetSlicedDataCubeMap(MWL_map_file, HI_sky_map, -27., -19.)
    PlotSkyMap(fig,'Intensity',logE_min,logE_max,HI_sky_map,f'{source_name}_HI_sky_map_m27_m19_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)
    GetSlicedDataCubeMap(MWL_map_file, HI_sky_map, -19., -3.)
    PlotSkyMap(fig,'Intensity',logE_min,logE_max,HI_sky_map,f'{source_name}_HI_sky_map_m19_m03_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)

if 'PSR_J1856_p0245' in source_name:
    HI_sky_map = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
    MWL_map_file = '/nevis/ged/data/rshang/MW_FITS/GALFA_HI_RA+DEC_284.00+02.35_N.fits' 
    GetSlicedDataCubeMapGALFA(MWL_map_file, HI_sky_map, 81.*1e3, 102.*1e3)
    PlotSkyMap(fig,'Intensity',logE_min,logE_max,HI_sky_map,f'{source_name}_HI_sky_map_p81_p102_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',zoomin=zoomin)



fig.clf()
figsize_x = 7
figsize_y = 7
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
axbig = fig.add_subplot()
label_x = 'elevation'
label_y = 'azimuth'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
for entry in range(0,len(list_run_elev)):
    axbig.scatter(list_run_elev[entry],list_run_azim[entry],color='b',alpha=0.5)
fig.savefig(f'output_plots/{source_name}_elevazim_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

fig.clf()
figsize_x = 6.4
figsize_y = 4.8
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
axbig = fig.add_subplot()
label_x = 'elevation [deg]'
label_y = 'number of runs'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.hist(list_run_elev, bins=20)
fig.savefig(f'output_plots/{source_name}_elev_{ana_tag}.png',bbox_inches='tight')
axbig.remove()


other_stars, other_star_type, other_star_coord = GetGammaSourceInfo() 
for star in range(0,len(other_stars)):
    if abs(src_ra-other_star_coord[star][0])>skymap_size: continue
    if abs(src_dec-other_star_coord[star][1])>skymap_size: continue
    print (f'Star {other_stars[star]} RA = {other_star_coord[star][0]:0.2f}, Dec = {other_star_coord[star][1]:0.2f}')

