
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
from common_functions import MyArray1D
from common_functions import MyArray3D

import common_functions

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
ReadRunListFromFile = common_functions.ReadRunListFromFile
build_skymap = common_functions.build_skymap
smooth_image = common_functions.smooth_image

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_input = os.environ.get("SMI_INPUT")
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")

source_name = sys.argv[1]
src_ra = float(sys.argv[2])
src_dec = float(sys.argv[3])
input_epoch = sys.argv[4] # 'V4', 'V5' or 'V6'

path_to_eigenvector = f'{smi_output}/eigenvectors_{source_name}_{input_epoch}.pkl'


data_xyoff_map = []
fit_xyoff_map = []
for logE in range(0,logE_bins):
    data_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    fit_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

on_runlist = ReadRunListFromFile(f'/nevis/tehanu/home/ryshang/veritas_analysis/easy-matrix-method/output_vts_hours/RunList_{source_name}_{input_epoch}.txt')

skymap_size = 3.
skymap_bins = 100
xsky_start = src_ra+skymap_size
xsky_end = src_ra-skymap_size
ysky_start = src_dec-skymap_size
ysky_end = src_dec+skymap_size

exposure_hours = 0.
data_sky_map = []
bkgd_sky_map = []
for logE in range(0,logE_bins):
    data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]

total_runs = len(on_runlist)
for run in range(0,total_runs):
    run_exposure_hours, run_all_sky_map, run_data_xyoff_map, run_fit_xyoff_map = build_skymap(smi_input,path_to_eigenvector,[on_runlist[run]],src_ra,src_dec)
    exposure_hours += run_exposure_hours
    for logE in range(0,logE_bins):
        data_xyoff_map[logE].add(run_data_xyoff_map[logE])
        fit_xyoff_map[logE].add(run_fit_xyoff_map[logE])
        for idx_x in range(0,skymap_bins):
            for idx_y in range(0,skymap_bins):
                data = run_all_sky_map[logE].waxis[idx_x,idx_y,0]
                bkg1 = run_all_sky_map[logE].waxis[idx_x,idx_y,1]
                bkg2 = run_all_sky_map[logE].waxis[idx_x,idx_y,2]
                bkg3 = run_all_sky_map[logE].waxis[idx_x,idx_y,3]
                #bkgd = (bkg1/1.+bkg2/2.+bkg3/3.)/(1.+1./2.+1./3.)
                bkgd = bkg1
                data_sky_map[logE].waxis[idx_x,idx_y,0] += data
                bkgd_sky_map[logE].waxis[idx_x,idx_y,0] += bkgd

    all_skymaps = [exposure_hours, data_sky_map, bkgd_sky_map, data_xyoff_map, fit_xyoff_map]
    output_filename = f'{smi_output}/skymaps_{source_name}_{input_epoch}.pkl'
    with open(output_filename,"wb") as file:
        pickle.dump(all_skymaps, file)




