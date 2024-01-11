
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.optimize import least_squares, minimize
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
matrix_rank = common_functions.matrix_rank
ReadRunListFromFile = common_functions.ReadRunListFromFile
build_big_camera_matrix = common_functions.build_big_camera_matrix
cosmic_ray_like_chi2 = common_functions.cosmic_ray_like_chi2

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_input = os.environ.get("SMI_INPUT")
smi_dir = os.environ.get("SMI_DIR")

on_matrix_xyoff_map = []
for logE in range(0,logE_bins):
    on_matrix_xyoff_map += [None]

print ('loading svd pickle data... ')
input_filename = f'{smi_dir}/output_eigenvector/eigenvectors.pkl'
big_eigenvectors = pickle.load(open(input_filename, "rb"))

data_xyoff_map = []
fit_xyoff_map = []
err_xyoff_map = []
for logE in range(0,logE_bins):
    data_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    fit_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    err_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

on_runlist = ReadRunListFromFile('../easy-matrix-method/output_vts_hours/RunList_CrabNebula_elev_60_70_V6.txt')
big_matrix = build_big_camera_matrix(smi_input,on_runlist,max_runs=1)

for logE in range(0,logE_bins):

    total_runs = len(big_matrix[logE])

    for run in range(0,total_runs):
        data_xyoff_map_1d = big_matrix[logE][run]
        init_params = big_eigenvectors[logE] @ data_xyoff_map_1d
    
        stepsize = [1e-3] * matrix_rank
        solution = minimize(
            cosmic_ray_like_chi2,
            x0=init_params,
            args=(big_eigenvectors[logE],data_xyoff_map_1d),
            method='L-BFGS-B',
            jac=None,
            options={'eps':stepsize,'ftol':0.001},
        )

        fit_params = solution['x']
        fit_xyoff_map_1d = big_eigenvectors[logE].T @ fit_params

        for gcut in range(0,gcut_bins):
            for idx_x in range(0,xoff_bins):
                for idx_y in range(0,yoff_bins):
                    idx_1d = gcut*xoff_bins*yoff_bins + idx_x*yoff_bins + idx_y
                    data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += data_xyoff_map_1d[idx_1d]
                    fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += fit_xyoff_map_1d[idx_1d]

    for gcut in range(0,gcut_bins):
        for idx_x in range(0,xoff_bins):
            for idx_y in range(0,yoff_bins):
                data = data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                model = fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                data_err = max(1.,pow(data,0.5))
                err_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = (data-model)/data_err

    for gcut in range(0,gcut_bins):
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'Xoff'
        label_y = 'Yoff'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = data_xyoff_map[logE].xaxis.min()
        xmax = data_xyoff_map[logE].xaxis.max()
        ymin = data_xyoff_map[logE].yaxis.min()
        ymax = data_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(data_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
        cbar = fig.colorbar(im)
        fig.savefig(f'output_plots/xyoff_map_logE{logE}_gcut{gcut}_data.png',bbox_inches='tight')
        axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'Xoff'
        label_y = 'Yoff'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = fit_xyoff_map[logE].xaxis.min()
        xmax = fit_xyoff_map[logE].xaxis.max()
        ymin = fit_xyoff_map[logE].yaxis.min()
        ymax = fit_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(fit_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
        cbar = fig.colorbar(im)
        fig.savefig(f'output_plots/xyoff_map_logE{logE}_gcut{gcut}_fit.png',bbox_inches='tight')
        axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'Xoff'
        label_y = 'Yoff'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = err_xyoff_map[logE].xaxis.min()
        xmax = err_xyoff_map[logE].xaxis.max()
        ymin = err_xyoff_map[logE].yaxis.min()
        ymax = err_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(err_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
        cbar = fig.colorbar(im)
        fig.savefig(f'output_plots/xyoff_map_logE{logE}_gcut{gcut}_err.png',bbox_inches='tight')
        axbig.remove()


