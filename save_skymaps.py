
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
from common_functions import MyArray1D
from common_functions import MyArray3D

import common_functions

logE_nbins = common_functions.logE_nbins
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
print (f'path_to_eigenvector = {path_to_eigenvector}')
path_to_eigenvector_ctl = f'{smi_output}/eigenvectors_ctl_{source_name}_{input_epoch}.pkl'


data_xyoff_map = []
fit_xyoff_map = []
data_xyoff_map_ctl = []
fit_xyoff_map_ctl = []
for logE in range(0,logE_nbins):
    data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    data_xyoff_map_ctl += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    fit_xyoff_map_ctl += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

on_runlist = ReadRunListFromFile(f'/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method/output_vts_query/RunList_{source_name}_{input_epoch}.txt')

skymap_size = 3.
skymap_bins = 100
xsky_start = src_ra+skymap_size
xsky_end = src_ra-skymap_size
ysky_start = src_dec-skymap_size
ysky_end = src_dec+skymap_size

exposure_hours = 0.
list_run_elev = []
list_run_azim = []
list_truth_params = []
list_fit_params = []
list_sr_chi2 = []
list_cr_chi2 = []

data_sky_map = []
bkgd_sky_map = []
for logE in range(0,logE_nbins):
    data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]


total_runs = len(on_runlist)
for run in range(0,total_runs):

    run_info, run_all_sky_map, run_data_xyoff_map, run_fit_xyoff_map = build_skymap(smi_input,path_to_eigenvector,[on_runlist[run]],src_ra,src_dec)
    run_info_ctl, run_all_sky_map_ctl, run_data_xyoff_map_ctl, run_fit_xyoff_map_ctl = build_skymap(smi_input,path_to_eigenvector_ctl,[on_runlist[run]],src_ra,src_dec,control_region=True)

    run_exposure_hours = run_info[0]
    run_elev = run_info[1]
    run_azim = run_info[2]
    truth_params = run_info[3]
    fit_params = run_info[4]
    sr_chi2 = run_info[5]
    cr_chi2 = run_info[6]
    if run_exposure_hours==0.: continue
    if run_elev<55.: continue
    exposure_hours += run_exposure_hours
    list_run_elev += [run_elev]
    list_run_azim += [run_azim]
    list_truth_params += [truth_params]
    list_fit_params += [fit_params]
    list_sr_chi2 += [sr_chi2]
    list_cr_chi2 += [cr_chi2]

    for logE in range(0,logE_nbins):
        data_xyoff_map[logE].add(run_data_xyoff_map[logE])
        fit_xyoff_map[logE].add(run_fit_xyoff_map[logE])
        data_xyoff_map_ctl[logE].add(run_data_xyoff_map_ctl[logE])
        fit_xyoff_map_ctl[logE].add(run_fit_xyoff_map_ctl[logE])

        run_fit_xyoff_sum_sr = np.sum(run_fit_xyoff_map[logE].waxis[:,:,0])
        run_fit_sky_sum_sr = 0.
        for gcut in range(1,gcut_bins):
            run_fit_sky_sum_sr += 1./float(gcut_bins-1)*np.sum(run_all_sky_map[logE].waxis[:,:,gcut])

        renormalization = 1.
        if run_fit_sky_sum_sr>0.:
            renormalization = run_fit_xyoff_sum_sr/run_fit_sky_sum_sr

        run_data_xyoff_sum_ctl = np.sum(run_data_xyoff_map_ctl[logE].waxis[:,:,0])
        run_fit_xyoff_sum_ctl = np.sum(run_fit_xyoff_map_ctl[logE].waxis[:,:,0])
        ctl_correction = 1.
        if run_fit_xyoff_sum_ctl>0.:
            ctl_correction = run_data_xyoff_sum_ctl/run_fit_xyoff_sum_ctl

        for idx_x in range(0,skymap_bins):
            for idx_y in range(0,skymap_bins):
                for gcut in range(1,gcut_bins):
                    if run_fit_sky_sum_sr==0.: continue
                    run_all_sky_map[logE].waxis[idx_x,idx_y,gcut] = run_all_sky_map[logE].waxis[idx_x,idx_y,gcut]*renormalization*ctl_correction

        for idx_x in range(0,skymap_bins):
            for idx_y in range(0,skymap_bins):
                data = run_all_sky_map[logE].waxis[idx_x,idx_y,0]
                data_sky_map[logE].waxis[idx_x,idx_y,0] += data
                bkgd = 0.
                for gcut in range(1,gcut_bins):
                    bkgd += 1./float(gcut_bins-1)*run_all_sky_map[logE].waxis[idx_x,idx_y,gcut]
                bkgd_sky_map[logE].waxis[idx_x,idx_y,0] += bkgd
                for gcut in range(1,gcut_bins):
                    bkgd_sky_map[logE].waxis[idx_x,idx_y,gcut] += run_all_sky_map[logE].waxis[idx_x,idx_y,gcut]

    for logE in range(0,logE_nbins):
        print ('=================================================================================')
        print (f'logE = {logE}')
        data_sum = np.sum(data_sky_map[logE].waxis[:,:,0])
        bkgd_sum = np.sum(bkgd_sky_map[logE].waxis[:,:,0])
        #print (f'Sky, data_sum = {data_sum}, bkgd_sum = {bkgd_sum:0.1f}')
        data_sum = np.sum(data_xyoff_map[logE].waxis[:,:,0])
        bkgd_sum = np.sum(fit_xyoff_map[logE].waxis[:,:,0])
        print (f'SR , data_sum = {data_sum}, bkgd_sum = {bkgd_sum:0.1f}')
        data_sum = np.sum(data_xyoff_map_ctl[logE].waxis[:,:,0])
        bkgd_sum = np.sum(fit_xyoff_map_ctl[logE].waxis[:,:,0])
        print (f'CR , data_sum = {data_sum}, bkgd_sum = {bkgd_sum:0.1f}')

    all_skymaps = [[exposure_hours, list_run_elev, list_run_azim, list_truth_params, list_fit_params, list_sr_chi2, list_cr_chi2], data_sky_map, bkgd_sky_map, data_xyoff_map, fit_xyoff_map]
    output_filename = f'{smi_output}/skymaps_{source_name}_{input_epoch}.pkl'
    with open(output_filename,"wb") as file:
        pickle.dump(all_skymaps, file)




