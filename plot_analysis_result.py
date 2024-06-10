
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
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
ReadRunListFromFile = common_functions.ReadRunListFromFile
build_skymap = common_functions.build_skymap
smooth_image = common_functions.smooth_image
PlotSkyMap = common_functions.PlotSkyMap
make_flux_map = common_functions.make_flux_map
make_significance_map = common_functions.make_significance_map
DefineRegionOfInterest = common_functions.DefineRegionOfInterest
PrintInformationRoI = common_functions.PrintInformationRoI
GetRadialProfile = common_functions.GetRadialProfile
plot_radial_profile_with_systematics = common_functions.plot_radial_profile_with_systematics
fit_2d_model = common_functions.fit_2d_model
matrix_rank = common_functions.matrix_rank
skymap_size = common_functions.skymap_size
skymap_bins = common_functions.skymap_bins
fine_skymap_bins = common_functions.fine_skymap_bins
GetGammaSourceInfo = common_functions.GetGammaSourceInfo
build_radial_symmetric_model = common_functions.build_radial_symmetric_model
doFluxCalibration = common_functions.doFluxCalibration
diffusion_func = common_functions.diffusion_func


fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_input = os.environ.get("SMI_INPUT")
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")

#ana_tag = 'linear'
#ana_tag = 'poisson'
#ana_tag = 'binspec'
ana_tag = 'fullspec'
#ana_tag = 'rank5'

qual_cut = 0.
#qual_cut = 20.

elev_cut = 20.
#elev_cut = 55.
cr_qual_cut = 1e10

#bias_array = [-0.023, -0.011, -0.022, -0.03,  -0.025,  0.018, -0.048, -0.378, -0.271]

source_name = sys.argv[1]
src_ra = float(sys.argv[2])
src_dec = float(sys.argv[3])
onoff = sys.argv[4]

logE_min = 0
logE_mid = 4
logE_max = logE_nbins
fit_radial_profile = False
make_symmetric_model = False
radial_bin_scale = 0.1

if 'Crab' in source_name:
    logE_min = 1
    logE_mid = 4
    logE_max = logE_nbins
    fit_radial_profile = False
    make_symmetric_model = False
if 'PSR_J1856_p0245' in source_name:
    logE_min = 2
    logE_mid = 5
    logE_max = logE_nbins
    fit_radial_profile = True
    make_symmetric_model = True
if 'PSR_J1907_p0602' in source_name:
    logE_min = 2
    logE_mid = 5
    logE_max = logE_nbins
    fit_radial_profile = True
    make_symmetric_model = False
if 'SS433' in source_name:
    logE_min = 0
    logE_mid = 4
    logE_max = logE_nbins
    fit_radial_profile = False
    make_symmetric_model = False
if 'PSR_J2021_p4026' in source_name:
    logE_min = 1
    logE_mid = 5
    logE_max = logE_nbins
    fit_radial_profile = False
    make_symmetric_model = False
if 'Geminga' in source_name:
    logE_min = 0
    logE_mid = 5
    logE_max = logE_nbins
    fit_radial_profile = True
    make_symmetric_model = False
    radial_bin_scale = 0.3

if doFluxCalibration:
    logE_min = 0
    logE_mid = 4
    logE_max = logE_nbins

#input_epoch = ['V4']
#input_epoch = ['V5']
#input_epoch = ['V6']
input_epoch = ['V4','V5','V6']

n_mimic = 0
#if onoff=='ON':
#    n_mimic = 5

xsky_start = src_ra+skymap_size
xsky_end = src_ra-skymap_size
ysky_start = src_dec-skymap_size
ysky_end = src_dec+skymap_size

if onoff=='ON':
    skymap_bins = fine_skymap_bins

region_name = source_name
if onoff=='OFF':
    region_name = 'Validation'
all_roi_name,all_roi_x,all_roi_y,all_roi_r = DefineRegionOfInterest(region_name,src_ra,src_dec)

total_exposure = 0.
good_exposure = 0.
list_run_elev = []
list_run_azim = []
list_truth_params = []
list_fit_params = []
list_sr_qual = []
list_cr_qual = []
sum_incl_sky_map = []
sum_data_sky_map = []
sum_bkgd_sky_map = []
sum_incl_sky_map_smooth = []
sum_data_sky_map_smooth = []
sum_bkgd_sky_map_smooth = []
sum_excess_sky_map_smooth = []
sum_significance_sky_map = []
sum_flux_sky_map = []
sum_flux_err_sky_map = []
sum_flux_sky_map_smooth = []
sum_flux_err_sky_map_smooth = []
for logE in range(0,logE_nbins):
    sum_incl_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_incl_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_data_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_bkgd_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_excess_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_significance_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_err_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_err_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
sum_data_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_data_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_data_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_bkgd_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_bkgd_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_bkgd_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_significance_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_significance_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_significance_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_excess_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_excess_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_excess_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_allE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_allE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_LE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_HE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_LE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_LE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_HE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_HE_smooth = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)

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
            sr_qual = run_info[5]
            cr_qual = run_info[6]

            if run_azim>270.:
                run_azim = run_azim-360.

            if run_elev<elev_cut:
                continue

            if not 'MIMIC' in mode:
                total_exposure += exposure

            is_good_run = True
            if cr_qual>cr_qual_cut:
                is_good_run = False
            if not is_good_run: 
                print (f'bad fitting. reject the run.')
                continue

            incl_sky_map = analysis_result[run][1] 
            data_sky_map = analysis_result[run][2] 
            bkgd_sky_map = analysis_result[run][3] 
            data_xyoff_map = analysis_result[run][4]
            fit_xyoff_map = analysis_result[run][5]
            ratio_xyoff_map = analysis_result[run][6]

            if not 'MIMIC' in mode:
                good_exposure += exposure
                list_run_elev += [run_elev]
                list_run_azim += [run_azim]
                list_truth_params += [truth_params]
                list_fit_params += [fit_params]
                list_sr_qual += [sr_qual]
                list_cr_qual += [cr_qual]
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
    for gcut in range(0,gcut_bins):
        for idx_x in range(0,xoff_bins[logE]):
            for idx_y in range(0,yoff_bins[logE]):
                data = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                model = sum_fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                data_err = max(1.,pow(data,0.5))
                sum_err_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = (data-model)/data_err

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
    smooth_size = 0.08
    #smooth_size = 0.12
    smooth_image(sum_incl_sky_map_smooth[logE].waxis[:,:,0],sum_incl_sky_map_smooth[logE].xaxis,sum_incl_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
    smooth_image(sum_bkgd_sky_map_smooth[logE].waxis[:,:,0],sum_bkgd_sky_map_smooth[logE].xaxis,sum_bkgd_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
    smooth_image(sum_data_sky_map_smooth[logE].waxis[:,:,0],sum_data_sky_map_smooth[logE].xaxis,sum_data_sky_map_smooth[logE].yaxis,kernel_radius=smooth_size)
    sum_data_sky_map_allE.add(sum_data_sky_map_smooth[logE])
    sum_bkgd_sky_map_allE.add(sum_bkgd_sky_map_smooth[logE])
    if logE>=logE_min and logE<logE_mid:
        sum_data_sky_map_LE.add(sum_data_sky_map_smooth[logE])
        sum_bkgd_sky_map_LE.add(sum_bkgd_sky_map_smooth[logE])
    if logE>=logE_mid and logE<=logE_max:
        sum_data_sky_map_HE.add(sum_data_sky_map_smooth[logE])
        sum_bkgd_sky_map_HE.add(sum_bkgd_sky_map_smooth[logE])

for logE in range(0,logE_nbins):
    for mimic in range(0,n_mimic):
        sum_mimic_incl_sky_map_smooth[mimic][logE].reset()
        sum_mimic_data_sky_map_smooth[mimic][logE].reset()
        sum_mimic_bkgd_sky_map_smooth[mimic][logE].reset()
        sum_mimic_excess_sky_map_smooth[mimic][logE].reset()
        sum_mimic_incl_sky_map_smooth[mimic][logE].add(sum_mimic_incl_sky_map[mimic][logE])
        sum_mimic_data_sky_map_smooth[mimic][logE].add(sum_mimic_data_sky_map[mimic][logE])
        sum_mimic_bkgd_sky_map_smooth[mimic][logE].add(sum_mimic_bkgd_sky_map[mimic][logE])
        smooth_size = 0.06
        #smooth_size = 0.08
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
    make_significance_map(sum_data_sky_map_smooth[logE],sum_bkgd_sky_map_smooth[logE],sum_significance_sky_map[logE],sum_excess_sky_map_smooth[logE])
make_significance_map(sum_data_sky_map_allE,sum_bkgd_sky_map_allE,sum_significance_sky_map_allE,sum_excess_sky_map_allE)
make_significance_map(sum_data_sky_map_LE,sum_bkgd_sky_map_LE,sum_significance_sky_map_LE,sum_excess_sky_map_LE)
make_significance_map(sum_data_sky_map_HE,sum_bkgd_sky_map_HE,sum_significance_sky_map_HE,sum_excess_sky_map_HE)

for mimic in range(0,n_mimic):
    make_significance_map(sum_mimic_data_sky_map_allE[mimic],sum_mimic_bkgd_sky_map_allE[mimic],sum_mimic_significance_sky_map_allE[mimic],sum_mimic_excess_sky_map_allE[mimic])

for logE in range(0,logE_nbins):
    avg_energy = 0.5*(pow(10.,logE_axis.xaxis[logE])+pow(10.,logE_axis.xaxis[logE+1]))
    delta_energy = 0.5*(pow(10.,logE_axis.xaxis[logE+1])-pow(10.,logE_axis.xaxis[logE]))
    make_flux_map(sum_incl_sky_map_smooth[logE],sum_data_sky_map_smooth[logE],sum_bkgd_sky_map_smooth[logE],sum_flux_sky_map_smooth[logE],sum_flux_err_sky_map_smooth[logE],avg_energy,delta_energy)
    make_flux_map(sum_incl_sky_map_smooth[logE],sum_data_sky_map[logE],sum_bkgd_sky_map[logE],sum_flux_sky_map[logE],sum_flux_err_sky_map[logE],avg_energy,delta_energy)

for mimic in range(0,n_mimic):
    for logE in range(0,logE_nbins):
        make_flux_map(sum_mimic_incl_sky_map_smooth[mimic][logE],sum_mimic_data_sky_map[mimic][logE],sum_mimic_bkgd_sky_map[mimic][logE],sum_mimic_flux_sky_map[mimic][logE],sum_mimic_flux_err_sky_map[mimic][logE],avg_energy,delta_energy)

for logE in range(logE_min,logE_max):
    PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE,logE+1,sum_flux_sky_map_smooth[logE],f'{source_name}_flux_sky_map_logE{logE}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
    PlotSkyMap(fig,'background count',logE,logE+1,sum_bkgd_sky_map_smooth[logE],f'{source_name}_bkgd_sky_map_logE{logE}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',layer=0)
    PlotSkyMap(fig,'excess count',logE,logE+1,sum_excess_sky_map_smooth[logE],f'{source_name}_excess_sky_map_logE{logE}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma',layer=0)
    sum_flux_sky_map_allE.add(sum_flux_sky_map[logE])
    sum_flux_err_sky_map_allE.addSquare(sum_flux_err_sky_map[logE])
    sum_flux_sky_map_allE_smooth.add(sum_flux_sky_map_smooth[logE])
    sum_flux_err_sky_map_allE_smooth.addSquare(sum_flux_err_sky_map_smooth[logE])
    for mimic in range(0,n_mimic):
        sum_mimic_flux_sky_map_allE[mimic].add(sum_mimic_flux_sky_map[mimic][logE])
        sum_mimic_flux_err_sky_map_allE[mimic].addSquare(sum_mimic_flux_err_sky_map[mimic][logE])
    if logE<logE_min: continue
    if logE>logE_max: continue
    if logE>=logE_min and logE<logE_mid:
        sum_flux_sky_map_LE.add(sum_flux_sky_map[logE])
        sum_flux_err_sky_map_LE.addSquare(sum_flux_err_sky_map[logE])
        sum_flux_sky_map_LE_smooth.add(sum_flux_sky_map_smooth[logE])
        sum_flux_err_sky_map_LE_smooth.addSquare(sum_flux_err_sky_map_smooth[logE])
        for mimic in range(0,n_mimic):
            sum_mimic_flux_sky_map_LE[mimic].add(sum_mimic_flux_sky_map[mimic][logE])
            sum_mimic_flux_err_sky_map_LE[mimic].addSquare(sum_mimic_flux_err_sky_map[mimic][logE])
    if logE>=logE_mid and logE<=logE_max:
        sum_flux_sky_map_HE.add(sum_flux_sky_map[logE])
        sum_flux_err_sky_map_HE.addSquare(sum_flux_err_sky_map[logE])
        sum_flux_sky_map_HE_smooth.add(sum_flux_sky_map_smooth[logE])
        sum_flux_err_sky_map_HE_smooth.addSquare(sum_flux_err_sky_map_smooth[logE])
        for mimic in range(0,n_mimic):
            sum_mimic_flux_sky_map_HE[mimic].add(sum_mimic_flux_sky_map[mimic][logE])
            sum_mimic_flux_err_sky_map_HE[mimic].addSquare(sum_mimic_flux_err_sky_map[mimic][logE])
PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE_min,logE_max,sum_flux_sky_map_allE_smooth,f'{source_name}_flux_sky_map_allE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE_min,logE_mid,sum_flux_sky_map_LE_smooth,f'{source_name}_flux_sky_map_LE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE_mid,logE_max,sum_flux_sky_map_HE_smooth,f'{source_name}_flux_sky_map_HE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')

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

for logE in range(logE_min,logE_max):
    low_energy = int(1000.*pow(10.,logE_bins[logE]))
    high_energy = int(1000.*pow(10.,logE_bins[logE+1]))
    SaveFITS(sum_flux_sky_map_smooth[logE],f'sum_flux_sky_map_{low_energy}GeV_{high_energy}GeV')
    SaveFITS(sum_flux_err_sky_map_smooth[logE],f'sum_flux_err_sky_map_{low_energy}GeV_{high_energy}GeV')

for roi in range(0,len(all_roi_name)):

    roi_name = all_roi_name[roi]
    roi_x = [all_roi_x[roi]]
    roi_y = [all_roi_y[roi]]
    roi_r = [all_roi_r[roi]]
    excl_roi_x = []
    excl_roi_y = []
    excl_roi_r = []

    PrintInformationRoI(fig,logE_min,logE_mid,logE_max,source_name,sum_data_sky_map,sum_bkgd_sky_map,sum_flux_sky_map,sum_flux_err_sky_map,sum_mimic_data_sky_map,sum_mimic_bkgd_sky_map,roi_name,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)


roi_name = all_roi_name[0]
roi_x = all_roi_x[0]
roi_y = all_roi_y[0]
roi_r = all_roi_r[0]
excl_roi_x = []
excl_roi_y = []
excl_roi_r = []

if make_symmetric_model:
    for roi2 in range(1,len(all_roi_name)):
        excl_roi_x += [all_roi_x[roi2]]
        excl_roi_y += [all_roi_y[roi2]]
        excl_roi_r += [all_roi_r[roi2]]

    radial_symmetry_sky_map = []
    for logE in range(0,logE_nbins):
        radial_map = MyArray3D()
        radial_map.just_like(sum_flux_sky_map[logE])
        radial_symmetry_sky_map += [radial_map]
    
for logE in range(logE_min,logE_max):
    on_radial_axis, on_profile_axis, on_profile_err_axis = GetRadialProfile(sum_flux_sky_map[logE],sum_flux_err_sky_map[logE],roi_x,roi_y,2.0,excl_roi_x,excl_roi_y,excl_roi_r)
    all_radial_axis, all_profile_axis, all_profile_err_axis = GetRadialProfile(sum_flux_sky_map[logE],sum_flux_err_sky_map[logE],roi_x,roi_y,2.0,excl_roi_x,excl_roi_y,excl_roi_r,use_excl=False)
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
    axbig.errorbar(all_radial_axis,all_profile_axis,all_profile_err_axis,color='r',marker='+',ls='none')
    axbig.errorbar(on_radial_axis,on_profile_axis,on_profile_err_axis,color='k',marker='+',ls='none')
    fig.savefig(f'output_plots/{source_name}_surface_brightness_logE{logE}_{roi_name}_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    if make_symmetric_model:
        build_radial_symmetric_model(radial_symmetry_sky_map[logE],on_radial_axis,on_profile_axis,roi_x,roi_y)
        
        PrintInformationRoI(fig,logE_min,logE_mid,logE_max,source_name,sum_data_sky_map,sum_bkgd_sky_map,radial_symmetry_sky_map,sum_flux_err_sky_map,sum_mimic_data_sky_map,sum_mimic_bkgd_sky_map,f'{roi_name}_symmetric',[roi_x],[roi_y],[roi_r],excl_roi_x,excl_roi_y,excl_roi_r)

        radial_symmetry_sky_map_allE = MyArray3D()
        radial_symmetry_sky_map_allE.just_like(radial_symmetry_sky_map[0])
        for logE in range(logE_min,logE_max):
            radial_symmetry_sky_map_allE.add(radial_symmetry_sky_map[logE])

        radial_symmetry_sky_map_LE = MyArray3D()
        radial_symmetry_sky_map_LE.just_like(radial_symmetry_sky_map[0])
        if logE>=logE_min and logE<logE_mid:
            radial_symmetry_sky_map_LE.add(radial_symmetry_sky_map[logE])

        radial_symmetry_sky_map_HE = MyArray3D()
        radial_symmetry_sky_map_HE.just_like(radial_symmetry_sky_map[0])
        if logE>=logE_mid and logE<=logE_max:
            radial_symmetry_sky_map_HE.add(radial_symmetry_sky_map[logE])

        PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE_min,logE_max,radial_symmetry_sky_map_allE,f'{source_name}_flux_sky_map_allE_symmetric_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')

for roi in range(0,len(all_roi_name)):

    roi_name = all_roi_name[roi]
    roi_x = all_roi_x[roi]
    roi_y = all_roi_y[roi]
    roi_r = all_roi_r[roi]
    excl_roi_x = []
    excl_roi_y = []
    excl_roi_r = []
    if roi==0:
        for roi2 in range(1,len(all_roi_name)):
            excl_roi_x += [all_roi_x[roi2]]
            excl_roi_y += [all_roi_y[roi2]]
            excl_roi_r += [all_roi_r[roi2]]

    fit_radial_profile_roi = False
    if roi==0:
        fit_radial_profile_roi = fit_radial_profile
    
    flux_sky_map = sum_flux_sky_map_allE
    if roi!=0 and make_symmetric_model:
        flux_sky_map.add(radial_symmetry_sky_map_allE,factor=-1.)
        PlotSkyMap(fig,'$E^{2}$ dN/dE [$\mathrm{TeV}\cdot\mathrm{cm}^{-2}\mathrm{s}^{-1}$]',logE_min,logE_max,flux_sky_map,f'{source_name}_flux_sky_map_allE_removal_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
    flux_err_sky_map = sum_flux_err_sky_map_allE
    mimic_flux_sky_map = sum_mimic_flux_sky_map_allE
    mimic_flux_err_sky_map = sum_mimic_flux_err_sky_map_allE
    plotname = f'{source_name}_surface_brightness_allE_{roi_name}_{ana_tag}'
    plot_radial_profile_with_systematics(fig,plotname,flux_sky_map,flux_err_sky_map,mimic_flux_sky_map,mimic_flux_err_sky_map,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,fit_radial_profile_roi,radial_bin_scale=radial_bin_scale)
    
    flux_sky_map = sum_flux_sky_map_LE
    if roi!=0 and make_symmetric_model:
        flux_sky_map.add(radial_symmetry_sky_map_LE,factor=-1.)
    flux_err_sky_map = sum_flux_err_sky_map_LE
    mimic_flux_sky_map = sum_mimic_flux_sky_map_LE
    mimic_flux_err_sky_map = sum_mimic_flux_err_sky_map_LE
    plotname = f'{source_name}_surface_brightness_LE_{roi_name}_{ana_tag}'
    plot_radial_profile_with_systematics(fig,plotname,flux_sky_map,flux_err_sky_map,mimic_flux_sky_map,mimic_flux_err_sky_map,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,fit_radial_profile_roi,radial_bin_scale=radial_bin_scale)
    
    flux_sky_map = sum_flux_sky_map_HE
    if roi!=0 and make_symmetric_model:
        flux_sky_map.add(radial_symmetry_sky_map_HE,factor=-1.)
    flux_err_sky_map = sum_flux_err_sky_map_HE
    mimic_flux_sky_map = sum_mimic_flux_sky_map_HE
    mimic_flux_err_sky_map = sum_mimic_flux_err_sky_map_HE
    plotname = f'{source_name}_surface_brightness_HE_{roi_name}_{ana_tag}'
    plot_radial_profile_with_systematics(fig,plotname,flux_sky_map,flux_err_sky_map,mimic_flux_sky_map,mimic_flux_err_sky_map,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r,fit_radial_profile_roi,radial_bin_scale=radial_bin_scale)

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
    fig.savefig(f'output_plots/{source_name}_surface_brightness_Fermi_{roi_name}_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

for logE in range(logE_min,logE_max):

    PlotSkyMap(fig,'significance',logE,logE+1,sum_significance_sky_map[logE],f'{source_name}_significance_sky_map_logE{logE}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=5.)

    max_z = 5.
    fig.clf()
    figsize_x = 7
    figsize_y = 7
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'Xoff'
    label_y = 'Yoff'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = sum_init_err_xyoff_map[logE].xaxis.min()
    xmax = sum_init_err_xyoff_map[logE].xaxis.max()
    ymin = sum_init_err_xyoff_map[logE].yaxis.min()
    ymax = sum_init_err_xyoff_map[logE].yaxis.max()
    im = axbig.imshow(sum_init_err_xyoff_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
    cbar = fig.colorbar(im)
    fig.savefig(f'output_plots/{source_name}_xyoff_init_err_map_logE{logE}_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    figsize_x = 7
    figsize_y = 7
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    axbig = fig.add_subplot()
    label_x = 'Xoff'
    label_y = 'Yoff'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = sum_ratio_xyoff_map[logE].xaxis.min()
    xmax = sum_ratio_xyoff_map[logE].xaxis.max()
    ymin = sum_ratio_xyoff_map[logE].yaxis.min()
    ymax = sum_ratio_xyoff_map[logE].yaxis.max()
    im = axbig.imshow(sum_ratio_xyoff_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    cbar = fig.colorbar(im)
    fig.savefig(f'output_plots/{source_name}_xyoff_map_logE{logE}_ratio_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    #for gcut in range(0,gcut_bins):
    #    fig.clf()
    #    figsize_x = 7
    #    figsize_y = 7
    #    fig.set_figheight(figsize_y)
    #    fig.set_figwidth(figsize_x)
    #    axbig = fig.add_subplot()
    #    label_x = 'Xoff'
    #    label_y = 'Yoff'
    #    axbig.set_xlabel(label_x)
    #    axbig.set_ylabel(label_y)
    #    xmin = sum_data_xyoff_map[logE].xaxis.min()
    #    xmax = sum_data_xyoff_map[logE].xaxis.max()
    #    ymin = sum_data_xyoff_map[logE].yaxis.min()
    #    ymax = sum_data_xyoff_map[logE].yaxis.max()
    #    im = axbig.imshow(sum_data_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    #    cbar = fig.colorbar(im)
    #    fig.savefig(f'output_plots/{source_name}_xyoff_map_logE{logE}_gcut{gcut}_data_{ana_tag}.png',bbox_inches='tight')
    #    axbig.remove()

    #    fig.clf()
    #    figsize_x = 7
    #    figsize_y = 7
    #    fig.set_figheight(figsize_y)
    #    fig.set_figwidth(figsize_x)
    #    axbig = fig.add_subplot()
    #    label_x = 'Xoff'
    #    label_y = 'Yoff'
    #    axbig.set_xlabel(label_x)
    #    axbig.set_ylabel(label_y)
    #    xmin = sum_err_xyoff_map[logE].xaxis.min()
    #    xmax = sum_err_xyoff_map[logE].xaxis.max()
    #    ymin = sum_err_xyoff_map[logE].yaxis.min()
    #    ymax = sum_err_xyoff_map[logE].yaxis.max()
    #    im = axbig.imshow(sum_err_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
    #    cbar = fig.colorbar(im)
    #    fig.savefig(f'output_plots/{source_name}_xyoff_err_map_logE{logE}_gcut{gcut}_{ana_tag}.png',bbox_inches='tight')
    #    axbig.remove()

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

PlotSkyMap(fig,'significance',logE_min,logE_max,sum_significance_sky_map_allE,f'{source_name}_significance_sky_map_allE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=5.)
PlotSkyMap(fig,'excess count',logE_min,logE_max,sum_excess_sky_map_allE,f'{source_name}_excess_sky_map_allE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
PlotSkyMap(fig,'significance',logE_min,logE_mid,sum_significance_sky_map_LE,f'{source_name}_significance_sky_map_LE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=5.)
PlotSkyMap(fig,'excess count',logE_min,logE_mid,sum_excess_sky_map_LE,f'{source_name}_excess_sky_map_LE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')
PlotSkyMap(fig,'significance',logE_mid,logE_max,sum_significance_sky_map_HE,f'{source_name}_significance_sky_map_HE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=5.)
PlotSkyMap(fig,'excess count',logE_mid,logE_max,sum_excess_sky_map_HE,f'{source_name}_excess_sky_map_HE_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,colormap='magma')

for mimic in range(0,n_mimic):
    PlotSkyMap(fig,'significance',logE_min,logE_max,sum_mimic_significance_sky_map_allE[mimic],f'{source_name}_significance_sky_map_allE_mimic{mimic}_{ana_tag}',roi_x=all_roi_x,roi_y=all_roi_y,roi_r=all_roi_r,max_z=5.)

print (f'total_exposure = {total_exposure}')
print (f'good_exposure = {good_exposure}')



fig.clf()
figsize_x = 7
figsize_y = 7
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
axbig = fig.add_subplot()
label_x = 'CR chi2'
label_y = 'SR chi2'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(list_cr_qual,list_sr_qual,color='b',alpha=0.5)
#axbig.set_xscale('log')
#axbig.set_ylim(0.,2.)
#axbig.set_xlim(0.5,1.5)
fig.savefig(f'output_plots/{source_name}_crsr_qual_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

fig.clf()
figsize_x = 7
figsize_y = 5
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
axbig = fig.add_subplot()
label_x = 'Run elevation'
label_y = 'SR chi2'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(list_run_elev,list_sr_qual,color='b',alpha=0.5)
fig.savefig(f'output_plots/{source_name}_elev_sr_qual_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

fig.clf()
figsize_x = 7
figsize_y = 5
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
axbig = fig.add_subplot()
label_x = 'Run azimuth'
label_y = 'SR chi2'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(list_run_azim,list_sr_qual,color='b',alpha=0.5)
fig.savefig(f'output_plots/{source_name}_azim_sr_qual_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

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
figsize_x = 7
figsize_y = 5
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
axbig = fig.add_subplot()
label_x = 'elevation'
axbig.set_xlabel(label_x)
axbig.hist(list_run_elev, bins=20)
fig.savefig(f'output_plots/{source_name}_elev_{ana_tag}.png',bbox_inches='tight')
axbig.remove()


other_stars, other_star_type, other_star_coord = GetGammaSourceInfo() 
for star in range(0,len(other_stars)):
    if abs(src_ra-other_star_coord[star][0])>skymap_size: continue
    if abs(src_dec-other_star_coord[star][1])>skymap_size: continue
    print (f'Star {other_stars[star]} RA = {other_star_coord[star][0]:0.1f}, Dec = {other_star_coord[star][1]:0.1f}')

