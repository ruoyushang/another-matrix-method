
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
#from memory_profiler import profile
from common_functions import MyArray1D
from common_functions import MyArray3D

import common_functions

use_fullspec = common_functions.use_fullspec
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
fine_skymap_bins = common_functions.fine_skymap_bins

#@profile

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_runlist = os.environ.get("SMI_RUNLIST")
smi_input = os.environ.get("SMI_INPUT")
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")
sky_tag = os.environ.get("SKY_TAG")

source_name = sys.argv[1]
src_ra = float(sys.argv[2])
src_dec = float(sys.argv[3])
onoff = sys.argv[4]
input_epoch = sys.argv[5] # 'V4', 'V5' or 'V6'

output_filename = f'{smi_output}/skymaps_{source_name}_{input_epoch}_{onoff}_{sky_tag}.pkl'
if os.path.exists(output_filename):
    print (f'{output_filename} exists, delete...')
    os.remove(output_filename)

path_to_eigenvector = f'{smi_output}/eigenvectors_{source_name}_{onoff}_{input_epoch}_{sky_tag}.pkl'
print (f'path_to_eigenvector = {path_to_eigenvector}')
path_to_big_matrix = f'{smi_output}/big_off_matrix_{source_name}_{onoff}_{input_epoch}.pkl'
print (f'path_to_big_matrix = {path_to_big_matrix}')

if onoff=='ON' or 'MIMIC' in onoff:
    skymap_bins = fine_skymap_bins

on_file = f'{smi_runlist}/RunList_{source_name}_{input_epoch}.txt'
off_file = f'{smi_runlist}/PairList_{source_name}_{input_epoch}.txt'
mimic_file = f'{smi_runlist}/ImposterList_{source_name}_{input_epoch}.txt'
on_runlist, off_runlist, mimic_runlist = ReadRunListFromFile(smi_input,on_file,off_file,mimic_file)

xsky_start = src_ra+skymap_size
xsky_end = src_ra-skymap_size
ysky_start = src_dec-skymap_size
ysky_end = src_dec+skymap_size



total_runs = len(on_runlist)
big_runlist = []
small_runlist = []
big_off_runlist = []
small_off_runlist = []
big_mimic_runlist = []
small_mimic_runlist = []

#nruns_in_small_list = 1
##nruns_in_small_list = 10
#run_count = 0
#for run in range(0,total_runs):
#    small_runlist += [on_runlist[run]]
#    small_off_runlist += [off_runlist[run]]
#    small_mimic_runlist += [mimic_runlist[run]]
#    run_count += 1
#    if (run % nruns_in_small_list)==(nruns_in_small_list-1):
#        big_runlist += [small_runlist]
#        small_runlist = []
#        big_off_runlist += [small_off_runlist]
#        small_off_runlist = []
#        big_mimic_runlist += [small_mimic_runlist]
#        small_mimic_runlist = []
#        run_count = 0

total_exposure = 0.
for run in range(0,total_runs):

    rootfile_name = f'{smi_input}/{on_runlist[run]}.anasum.root'
    print (rootfile_name)
    if not os.path.exists(rootfile_name):
        print (f'file does not exist.')
        continue
    InputFile = ROOT.TFile(rootfile_name)
    TreeName = f'run_{on_runlist[run]}/stereo/DL3EventTree'
    EvtTree = InputFile.Get(TreeName)
    total_entries = EvtTree.GetEntries()
    EvtTree.GetEntry(0)
    time_start = EvtTree.timeOfDay
    EvtTree.GetEntry(total_entries-1)
    time_end = EvtTree.timeOfDay

    total_exposure += (time_end-time_start)/3600.

#min_exposure = 0.1 # hours
min_exposure = 2.0 # hours
#min_exposure = 5.0 # hours
run_exposure = 0.
for run in range(0,total_runs):

    rootfile_name = f'{smi_input}/{on_runlist[run]}.anasum.root'
    print (rootfile_name)
    if not os.path.exists(rootfile_name):
        print (f'file does not exist.')
        if run==total_runs-1:
            big_runlist += [small_runlist]
            small_runlist = []
            big_off_runlist += [small_off_runlist]
            small_off_runlist = []
            big_mimic_runlist += [small_mimic_runlist]
            small_mimic_runlist = []
            run_exposure = 0.
        continue

    InputFile = ROOT.TFile(rootfile_name)
    TreeName = f'run_{on_runlist[run]}/stereo/DL3EventTree'
    EvtTree = InputFile.Get(TreeName)
    total_entries = EvtTree.GetEntries()
    EvtTree.GetEntry(0)
    time_start = EvtTree.timeOfDay
    EvtTree.GetEntry(total_entries-1)
    time_end = EvtTree.timeOfDay

    small_runlist += [on_runlist[run]]
    small_off_runlist += [off_runlist[run]]
    small_mimic_runlist += [mimic_runlist[run]]
    run_exposure += (time_end-time_start)/3600.

    if run==total_runs-1 or run_exposure>min_exposure:
        big_runlist += [small_runlist]
        small_runlist = []
        big_off_runlist += [small_off_runlist]
        small_off_runlist = []
        big_mimic_runlist += [small_mimic_runlist]
        small_mimic_runlist = []
        run_exposure = 0.

print (f'min_exposure = {min_exposure}')
print (f'len(big_runlist) = {len(big_runlist)}')

total_data_xyoff_map = []
total_fit_xyoff_map = []
for logE in range(0,logE_nbins):
    total_data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    total_fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

total_data_sky_map = []
total_bkgd_sky_map = []
total_syst_sky_map = []
for logE in range(0,logE_nbins):
    total_data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    total_bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    total_syst_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]

incl_sky_map = []
data_sky_map = []
bkgd_sky_map = []
syst_sky_map = []
for logE in range(0,logE_nbins):
    incl_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    syst_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]

data_xyoff_map = []
fit_xyoff_map = []
ratio_xyoff_map = []
for logE in range(0,logE_nbins):
    data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    ratio_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

run_incl_sky_map = []
run_data_sky_map = []
run_fit_sky_map = []
run_syst_sky_map = []
for logE in range(0,logE_nbins):
    run_incl_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    run_data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    run_fit_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    run_syst_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]

run_data_xyoff_map = []
run_fit_xyoff_map = []
run_tolerance_xyoff_map = []
run_ratio_xyoff_map = []
run_syst_xyoff_map = []
for logE in range(0,logE_nbins):
    run_data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    run_fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    run_tolerance_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    run_ratio_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    run_syst_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]


run_list_count = 0
for small_runlist in range(0,len(big_runlist)):

    for logE in range(0,logE_nbins):
        incl_sky_map[logE].reset()
        data_sky_map[logE].reset()
        bkgd_sky_map[logE].reset()
        syst_sky_map[logE].reset()

    for logE in range(0,logE_nbins):
        data_xyoff_map[logE].reset()
        fit_xyoff_map[logE].reset()
        ratio_xyoff_map[logE].reset()

    run_list_count += 1
    print (f'analyzing {run_list_count}/{len(big_runlist)} lists...')

    run_info = build_skymap(
            source_name,
            src_ra,
            src_dec,
            smi_input,
            path_to_eigenvector,
            path_to_big_matrix,
            big_runlist[small_runlist],
            big_mimic_runlist[small_runlist],
            onoff, 
            run_incl_sky_map, 
            run_data_sky_map, 
            run_fit_sky_map, 
            run_syst_sky_map, 
            run_data_xyoff_map, 
            run_fit_xyoff_map, 
            run_tolerance_xyoff_map, 
            run_ratio_xyoff_map,
            run_syst_xyoff_map,
        )

    run_exposure_hours = run_info[0]
    run_elev = run_info[1]
    run_azim = run_info[2]
    truth_params = run_info[3]
    fit_params = run_info[4]
    run_nsb = run_info[5]
    if run_exposure_hours==0.: continue

    run_data_xyoff_sum_sr = 0.
    run_data_sky_sum_sr = 0.
    run_fit_xyoff_sum_sr = 0.
    run_fit_sky_sum_sr = 0.
    for logE in range(0,logE_nbins):
        run_data_xyoff_sum_sr += np.sum(run_data_xyoff_map[logE].waxis[:,:,0])
        run_data_sky_sum_sr += np.sum(run_data_sky_map[logE].waxis[:,:,0])
        run_fit_xyoff_sum_sr += np.sum(run_fit_xyoff_map[logE].waxis[:,:,0])
        run_fit_sky_sum_sr += np.sum(run_fit_sky_map[logE].waxis[:,:,0])
    print (f'run_data_xyoff_sum_sr = {run_data_xyoff_sum_sr:0.1f}, run_data_sky_sum_sr = {run_data_sky_sum_sr:0.1f}')
    print (f'run_fit_xyoff_sum_sr = {run_fit_xyoff_sum_sr:0.1f}, run_fit_sky_sum_sr = {run_fit_sky_sum_sr:0.1f}')

    for logE in range(0,logE_nbins):
        data_xyoff_map[logE].add(run_data_xyoff_map[logE])
        fit_xyoff_map[logE].add(run_fit_xyoff_map[logE])
        ratio_xyoff_map[logE].add(run_ratio_xyoff_map[logE])
        total_data_xyoff_map[logE].add(run_data_xyoff_map[logE])
        total_fit_xyoff_map[logE].add(run_fit_xyoff_map[logE])

        for gcut in range(0,1):
            for idx_x in range(0,skymap_bins):
                for idx_y in range(0,skymap_bins):
                    incl_data = run_incl_sky_map[logE].waxis[idx_x,idx_y,gcut]
                    incl_sky_map[logE].waxis[idx_x,idx_y,gcut] += incl_data
                    data = run_data_sky_map[logE].waxis[idx_x,idx_y,gcut]
                    data_sky_map[logE].waxis[idx_x,idx_y,gcut] += data
                    bkgd = run_fit_sky_map[logE].waxis[idx_x,idx_y,gcut]
                    syst = run_syst_sky_map[logE].waxis[idx_x,idx_y,gcut]
                    bkgd_sky_map[logE].waxis[idx_x,idx_y,gcut] += bkgd
                    syst_sky_map[logE].waxis[idx_x,idx_y,gcut] += syst
                    total_data_sky_map[logE].waxis[idx_x,idx_y,gcut] += data
                    total_bkgd_sky_map[logE].waxis[idx_x,idx_y,gcut] += bkgd
                    total_syst_sky_map[logE].waxis[idx_x,idx_y,gcut] += syst

    print ('=================================================================================')
    total_data_sum = 0.
    total_bkgd_sum = 0.
    for logE in range(0,logE_nbins):
        print (f'logE = {logE}')
        data_sum = np.sum(total_data_sky_map[logE].waxis[:,:,0])
        bkgd_sum = np.sum(total_bkgd_sky_map[logE].waxis[:,:,0])
        syst_sum = np.sum(total_syst_sky_map[logE].waxis[:,:,0])
        total_data_sum += data_sum
        total_bkgd_sum += bkgd_sum
        error = 0.
        stat_error = 0.
        syst_error = 0.
        if data_sum>0.:
            error = 100.*(data_sum-bkgd_sum)/data_sum
            stat_error = 100.*pow(data_sum,0.5)/data_sum
            syst_error = 100.*pow(syst_sum,0.5)/data_sum
        print (f'On data,  data_sum = {data_sum}, bkgd_sum = {bkgd_sum:0.1f}, error = {error:0.1f} +/- {stat_error:0.1f} +/- {syst_error:0.1f} %')
    print (f'total_data_sum = {total_data_sum:0.1f}, total_bkgd_sum = {total_bkgd_sum:0.1f}')


    output_filename = f'{smi_output}/skymaps_{source_name}_{input_epoch}_{onoff}_{sky_tag}.pkl'
    print (f'reading {output_filename}...')
    if not os.path.exists(output_filename):
        print (f'{output_filename} does not exist, create new...')
        analysis_result = []
        analysis_result += [[[run_exposure_hours, run_elev, run_azim, truth_params, fit_params, run_nsb], incl_sky_map, data_sky_map, bkgd_sky_map, syst_sky_map, data_xyoff_map, fit_xyoff_map, ratio_xyoff_map]]
        with open(output_filename,"wb") as file:
            pickle.dump(analysis_result, file)
        del analysis_result
    else:
        analysis_result = pickle.load(open(output_filename, "rb"))
        analysis_result += [[[run_exposure_hours, run_elev, run_azim, truth_params, fit_params, run_nsb], incl_sky_map, data_sky_map, bkgd_sky_map, syst_sky_map, data_xyoff_map, fit_xyoff_map, ratio_xyoff_map]]
        with open(output_filename,"wb") as file:
            pickle.dump(analysis_result, file)
        del analysis_result
    print ('=================================================================================')

#output_filename = f'{smi_output}/skymaps_{source_name}_{input_epoch}_{onoff}_{sky_tag}.pkl'
#with open(output_filename,"wb") as file:
#    pickle.dump(all_skymaps, file)

#exit()


