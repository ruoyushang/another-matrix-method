
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
skymap_size = common_functions.skymap_size
skymap_bins = common_functions.skymap_bins

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


on_file = f'/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method/output_vts_query/RunList_{source_name}_{input_epoch}.txt'
off_file = f'/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method/output_vts_query/PairList_{source_name}_{input_epoch}.txt'
on_runlist, off_runlist = ReadRunListFromFile(on_file,off_file)

xsky_start = src_ra+skymap_size
xsky_end = src_ra-skymap_size
ysky_start = src_dec-skymap_size
ysky_end = src_dec+skymap_size



total_runs = len(on_runlist)
big_runlist = []
small_runlist = []
nruns_in_small_list = 1
#nruns_in_small_list = 20
run_count = 0
for run in range(0,total_runs):
    #if len(off_runlist[run])<2: 
    #    print (f'ON run {on_runlist[run]} rejected: zero matched OFF run')
    #    continue
    small_runlist += [on_runlist[run]]
    run_count += 1
    if (run % nruns_in_small_list)==0 and run_count>=nruns_in_small_list:
        big_runlist += [small_runlist]
        small_runlist = []
        run_count = 0

total_data_xyoff_map = []
total_fit_xyoff_map = []
for logE in range(0,logE_nbins):
    total_data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    total_fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]


run_list_count = 0
all_skymaps = []
for small_runlist in big_runlist:

    data_sky_map = []
    bkgd_sky_map = []
    for logE in range(0,logE_nbins):
        data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
        bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

    data_xyoff_map = []
    fit_xyoff_map = []
    for logE in range(0,logE_nbins):
        data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
        fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]


    run_list_count += 1
    print (f'analyzing {run_list_count}/{len(big_runlist)} lists...')

    run_info, run_data_sky_map, run_fit_sky_map, run_data_xyoff_map, run_fit_xyoff_map = build_skymap(smi_input,path_to_eigenvector,small_runlist,src_ra,src_dec)

    run_exposure_hours = run_info[0]
    run_elev = run_info[1]
    run_azim = run_info[2]
    truth_params = run_info[3]
    fit_params = run_info[4]
    sr_qual = run_info[5]
    cr_qual = run_info[6]
    if run_exposure_hours==0.: continue

    for logE in range(0,logE_nbins):
        data_xyoff_map[logE].add(run_data_xyoff_map[logE])
        fit_xyoff_map[logE].add(run_fit_xyoff_map[logE])
        total_data_xyoff_map[logE].add(run_data_xyoff_map[logE])
        total_fit_xyoff_map[logE].add(run_fit_xyoff_map[logE])

        run_data_xyoff_sum_cr = np.sum(run_data_xyoff_map[logE].waxis[:,:,1])
        run_data_sky_sum_cr = np.sum(run_data_sky_map[logE].waxis[:,:,1])
        run_fit_xyoff_sum_sr = np.sum(run_fit_xyoff_map[logE].waxis[:,:,0])
        run_fit_sky_sum_sr = 0.
        for gcut in range(1,gcut_bins):
            run_fit_sky_sum_sr += 1./float(gcut_bins-1)*np.sum(run_fit_sky_map[logE].waxis[:,:,gcut])

        renormalization = 1.
        if run_data_xyoff_sum_cr>0. and run_fit_sky_sum_sr>0.:
            renormalization = run_data_sky_sum_cr/run_data_xyoff_sum_cr*run_fit_xyoff_sum_sr/run_fit_sky_sum_sr
            #renormalization = run_fit_xyoff_sum_sr/run_fit_sky_sum_sr

        for idx_x in range(0,skymap_bins):
            for idx_y in range(0,skymap_bins):
                for gcut in range(1,gcut_bins):
                    run_fit_sky_map[logE].waxis[idx_x,idx_y,gcut] = run_fit_sky_map[logE].waxis[idx_x,idx_y,gcut]*renormalization

        for idx_x in range(0,skymap_bins):
            for idx_y in range(0,skymap_bins):
                data = run_data_sky_map[logE].waxis[idx_x,idx_y,0]
                data_sky_map[logE].waxis[idx_x,idx_y,0] += data
                bkgd = 0.
                for gcut in range(1,gcut_bins):
                    bkgd += 1./float(gcut_bins-1)*run_fit_sky_map[logE].waxis[idx_x,idx_y,gcut]
                bkgd_sky_map[logE].waxis[idx_x,idx_y,0] += bkgd
                for gcut in range(1,gcut_bins):
                    bkgd_sky_map[logE].waxis[idx_x,idx_y,gcut] += run_fit_sky_map[logE].waxis[idx_x,idx_y,gcut]

    print ('=================================================================================')
    for logE in range(0,logE_nbins):
        print (f'logE = {logE}')
        data_sum = np.sum(data_sky_map[logE].waxis[:,:,0])
        bkgd_sum = np.sum(bkgd_sky_map[logE].waxis[:,:,0])
        #data_sum = np.sum(total_data_xyoff_map[logE].waxis[:,:,0])
        #bkgd_sum = np.sum(total_fit_xyoff_map[logE].waxis[:,:,0])
        error = 0.
        stat_error = 0.
        if data_sum>0.:
            error = 100.*(data_sum-bkgd_sum)/data_sum
            stat_error = 100.*pow(data_sum,0.5)/data_sum
        print (f'On data,  data_sum = {data_sum}, bkgd_sum = {bkgd_sum:0.1f}, error = {error:0.1f} +/- {stat_error:0.1f} %')


    all_skymaps += [[[run_exposure_hours, run_elev, run_azim, truth_params, fit_params, sr_qual, cr_qual], data_sky_map, bkgd_sky_map, data_xyoff_map, fit_xyoff_map]]

    output_filename = f'{smi_output}/skymaps_{source_name}_{input_epoch}.pkl'
    with open(output_filename,"wb") as file:
        pickle.dump(all_skymaps, file)




