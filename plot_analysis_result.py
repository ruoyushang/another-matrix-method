
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
#input_epoch = ['V4']
input_epoch = ['V4','V5','V6']

skymap_size = 3.
skymap_bins = 100
xsky_start = src_ra-skymap_size
xsky_end = src_ra+skymap_size
ysky_start = src_dec-skymap_size
ysky_end = src_dec+skymap_size

sum_data_sky_map = []
sum_bkgd_sky_map = []
sum_data_sky_map_smooth = []
sum_bkgd_sky_map_smooth = []
sum_diff_sky_map = []
for logE in range(0,logE_bins):
    sum_data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_data_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_bkgd_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_diff_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
sum_data_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_bkgd_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_diff_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)

sum_data_xyoff_map = []
sum_fit_xyoff_map = []
sum_err_xyoff_map = []
sum_init_err_xyoff_map = []
for logE in range(0,logE_bins):
    sum_data_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_fit_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_err_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_init_err_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

for epoch in input_epoch:

    input_filename = f'{smi_output}/skymaps_{source_name}_{epoch}.pkl'
    print (f'reading {input_filename}...')
    if not os.path.exists(input_filename):
        print (f'{input_filename} does not exist.')
        continue
    analysis_result = pickle.load(open(input_filename, "rb"))
    
    data_sky_map = analysis_result[0] 
    bkgd_sky_map = analysis_result[1] 
    data_xyoff_map = analysis_result[2]
    fit_xyoff_map = analysis_result[3]

    for logE in range(0,logE_bins):
        sum_data_sky_map[logE].add(data_sky_map[logE])
        sum_bkgd_sky_map[logE].add(bkgd_sky_map[logE])
        sum_data_xyoff_map[logE].add(data_xyoff_map[logE])
        sum_fit_xyoff_map[logE].add(fit_xyoff_map[logE])
    
for logE in range(0,logE_bins):
    data_integral = 0.
    model_integral = 0.
    for idx_x in range(0,xoff_bins):
        for idx_y in range(0,yoff_bins):
            data_integral += sum_data_xyoff_map[logE].waxis[idx_x,idx_y,0] 
            model_integral += sum_data_xyoff_map[logE].waxis[idx_x,idx_y,1] 
    for idx_x in range(0,xoff_bins):
        for idx_y in range(0,yoff_bins):
            data = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,0] 
            model = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,1]*data_integral/model_integral 
            data_err = max(1.,pow(data,0.5))
            sum_init_err_xyoff_map[logE].waxis[idx_x,idx_y,0] = (data-model)/data_err

for logE in range(0,logE_bins):
    for gcut in range(0,gcut_bins):
        for idx_x in range(0,xoff_bins):
            for idx_y in range(0,yoff_bins):
                data = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                model = sum_fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                data_err = max(1.,pow(data,0.5))
                sum_err_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = (data-model)/data_err

    sum_data_sky_map_smooth[logE].reset()
    sum_bkgd_sky_map_smooth[logE].reset()
    sum_data_sky_map_smooth[logE].add(sum_data_sky_map[logE])
    sum_bkgd_sky_map_smooth[logE].add(sum_bkgd_sky_map[logE])
    smooth_image(sum_data_sky_map_smooth[logE].waxis[:,:,0],sum_data_sky_map_smooth[logE].xaxis,sum_data_sky_map_smooth[logE].yaxis)
    smooth_image(sum_bkgd_sky_map_smooth[logE].waxis[:,:,0],sum_bkgd_sky_map_smooth[logE].xaxis,sum_bkgd_sky_map_smooth[logE].yaxis)
    for idx_x in range(0,skymap_bins):
        for idx_y in range(0,skymap_bins):
            data = sum_data_sky_map_smooth[logE].waxis[idx_x,idx_y,0]
            bkgd = sum_bkgd_sky_map_smooth[logE].waxis[idx_x,idx_y,0]
            data_err = max(1.,pow(data,0.5))
            sum_diff_sky_map[logE].waxis[idx_x,idx_y,0] = (data-bkgd)/data_err
    sum_data_sky_map_allE.add(sum_data_sky_map_smooth[logE])
    sum_bkgd_sky_map_allE.add(sum_bkgd_sky_map_smooth[logE])

for idx_x in range(0,skymap_bins):
    for idx_y in range(0,skymap_bins):
        data = sum_data_sky_map_allE.waxis[idx_x,idx_y,0]
        bkgd = sum_bkgd_sky_map_allE.waxis[idx_x,idx_y,0]
        data_err = max(1.,pow(data,0.5))
        sum_diff_sky_map_allE.waxis[idx_x,idx_y,0] = (data-bkgd)/data_err

for logE in range(0,logE_bins):
    max_z = 5.
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'RA'
    label_y = 'Dec'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = sum_diff_sky_map[logE].xaxis.min()
    xmax = sum_diff_sky_map[logE].xaxis.max()
    ymin = sum_diff_sky_map[logE].yaxis.min()
    ymax = sum_diff_sky_map[logE].yaxis.max()
    im = axbig.imshow(sum_diff_sky_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
    cbar = fig.colorbar(im)
    fig.savefig(f'output_plots/{source_name}_diff_sky_map_logE{logE}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
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
    fig.savefig(f'output_plots/{source_name}_init_err_xyoff_map_logE{logE}.png',bbox_inches='tight')
    axbig.remove()

    for gcut in range(0,gcut_bins):
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'Xoff'
        label_y = 'Yoff'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = sum_data_xyoff_map[logE].xaxis.min()
        xmax = sum_data_xyoff_map[logE].xaxis.max()
        ymin = sum_data_xyoff_map[logE].yaxis.min()
        ymax = sum_data_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(sum_data_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
        cbar = fig.colorbar(im)
        fig.savefig(f'output_plots/{source_name}_xyoff_map_logE{logE}_gcut{gcut}_data.png',bbox_inches='tight')
        axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'Xoff'
        label_y = 'Yoff'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = sum_fit_xyoff_map[logE].xaxis.min()
        xmax = sum_fit_xyoff_map[logE].xaxis.max()
        ymin = sum_fit_xyoff_map[logE].yaxis.min()
        ymax = sum_fit_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(sum_fit_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
        cbar = fig.colorbar(im)
        fig.savefig(f'output_plots/{source_name}_xyoff_map_logE{logE}_gcut{gcut}_fit.png',bbox_inches='tight')
        axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'Xoff'
        label_y = 'Yoff'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = sum_err_xyoff_map[logE].xaxis.min()
        xmax = sum_err_xyoff_map[logE].xaxis.max()
        ymin = sum_err_xyoff_map[logE].yaxis.min()
        ymax = sum_err_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(sum_err_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
        cbar = fig.colorbar(im)
        fig.savefig(f'output_plots/{source_name}_err_xyoff_map_logE{logE}_gcut{gcut}.png',bbox_inches='tight')
        axbig.remove()

max_z = 5.
fig.clf()
axbig = fig.add_subplot()
label_x = 'RA'
label_y = 'Dec'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
xmin = sum_diff_sky_map_allE.xaxis.min()
xmax = sum_diff_sky_map_allE.xaxis.max()
ymin = sum_diff_sky_map_allE.yaxis.min()
ymax = sum_diff_sky_map_allE.yaxis.max()
im = axbig.imshow(sum_diff_sky_map_allE.waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
cbar = fig.colorbar(im)
fig.savefig(f'output_plots/{source_name}_diff_sky_map_allE.png',bbox_inches='tight')
axbig.remove()


