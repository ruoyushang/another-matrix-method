
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares, minimize
from common_functions import MyArray1D
from common_functions import MyArray3D

import common_functions

logE_bins = common_functions.logE_bins
matrix_rank = common_functions.matrix_rank
matrix_rank_fullspec = common_functions.matrix_rank_fullspec
ReadOffRunListFromFile = common_functions.ReadOffRunListFromFile
build_big_camera_matrix = common_functions.build_big_camera_matrix
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

fig, ax = plt.subplots()
figsize_x = 6.4
figsize_y = 4.6
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_input = os.environ.get("SMI_INPUT")
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")
sky_tag = os.environ.get("SKY_TAG")

source_name = sys.argv[1]
onoff = sys.argv[2]
input_epoch = sys.argv[3] # 'V4', 'V5' or 'V6'

print ('loading matrix pickle data... ')
input_filename = f'{smi_output}/big_off_matrix_{source_name}_{onoff}_{input_epoch}.pkl'
print (f'input_filename = {input_filename}')
big_matrix_pkl = pickle.load(open(input_filename, "rb"))
big_matrix = big_matrix_pkl[0]
big_matrix_fullspec = big_matrix_pkl[1]

for logE in range(0,logE_nbins):
    n_runs = float(len(big_matrix[logE]))
    big_matrix[logE] = np.array(big_matrix[logE])*1./n_runs

n_runs = float(len(big_matrix_fullspec))
big_matrix_fullspec = np.array(big_matrix_fullspec)*1./n_runs

print ('Computing SVD eigenvectors...')
big_xyoff_map_1d = []
for logE in range(0,logE_nbins):
    big_xyoff_map_1d += [np.zeros_like(big_matrix[logE][0])]

big_xyoff_map_1d_fullspec = np.zeros_like(big_matrix_fullspec[0])

for logE in range(0,logE_nbins):
    for entry in range(0,len(big_matrix[logE])):
        for pix in range(0,len(big_matrix[logE][entry])):
            big_xyoff_map_1d[logE][pix] += big_matrix[logE][entry][pix]

for entry in range(0,len(big_matrix_fullspec)):
    for pix in range(0,len(big_matrix_fullspec[entry])):
        big_xyoff_map_1d_fullspec[pix] += big_matrix_fullspec[entry][pix]

full_eigenvalues = []
big_eigenvalues = []
big_eigenvectors = []
for logE in range(0,logE_nbins):

    U_full, S_full, VT_full = np.linalg.svd(big_matrix[logE],full_matrices=False) # perform better for perturbation method
    print (f'S_full length = {len(S_full)}')

    avg_n_events = np.sum(big_matrix[logE])
    density_events = avg_n_events/float(len(S_full))
    print (f'density_events = {density_events}')

    effective_matrix_rank = min(matrix_rank,int(0.5*3./4.*(len(S_full)-1)))
    #if density_events<1.0:
    #    effective_matrix_rank = 1
    print (f'effective_matrix_rank = {effective_matrix_rank}')

    U_eco = U_full[:, :effective_matrix_rank]
    VT_eco = VT_full[:effective_matrix_rank, :]
    S_eco = S_full[:effective_matrix_rank]
    big_eigenvalues += [S_eco]
    big_eigenvectors += [VT_eco]
    full_eigenvalues += [S_full]
    
    rank_index = []
    for entry in range(0,len(S_full)):
        rank_index += [entry+1]
    
    
fig.clf()
axbig = fig.add_subplot()
label_x = '$k$'
label_y = '$\sigma_{k}$'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
plot_max_rank = min(int(0.5*len(full_eigenvalues[0])),300)
axbig.set_xlim(1,plot_max_rank)
#axbig.set_ylim(full_eigenvalues[0][plot_max_rank-1],2.*full_eigenvalues[0][0])
axbig.set_ylim(bottom=full_eigenvalues[0][plot_max_rank-1])
axbig.set_xscale('log')
axbig.set_yscale('log')
for logE in range(0,logE_nbins):
    E_min = pow(10.,logE_bins[logE])
    E_max = pow(10.,logE_bins[logE+1])
    axbig.plot(full_eigenvalues[logE],label=f'E = {E_min:0.2f} - {E_max:0.2f} TeV')
axbig.legend(loc='best')
fig.savefig(f'{smi_dir}/output_plots/signularvalue_{source_name}_{input_epoch}.png',bbox_inches='tight')
axbig.remove()
    

max_matrix_rank = min(5,big_eigenvectors[0].shape[0])
for rank in range(0,max_matrix_rank):

    list_eigen_xyoff_map = []
    for logE in range(0,logE_nbins):
        eigen_xyoff_map = MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)
        list_eigen_xyoff_map += [eigen_xyoff_map]

    for logE in range(0,logE_nbins):
        idx_1d = 0
        for gcut in range(0,gcut_bins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d += 1
                    list_eigen_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += big_eigenvectors[logE][rank][idx_1d-1]

    fig.clf()
    figsize_x = 2.*gcut_bins
    figsize_y = 2.*logE_nbins
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    ax_idx = 0
    for logE in range(0,logE_nbins):
        for gcut in range(0,gcut_bins):
            ax_idx = gcut + (logE-0)*gcut_bins + 1
            axbig = fig.add_subplot(logE_nbins,gcut_bins,ax_idx)
            if logE==0:
                if gcut==0:
                    axbig.set_title('SR')
                else:
                    axbig.set_title(f'CR{gcut}')
            if gcut==0:
                axbig.set_ylabel(f'{pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
            xmin = list_eigen_xyoff_map[logE].xaxis.min()
            xmax = list_eigen_xyoff_map[logE].xaxis.max()
            ymin = list_eigen_xyoff_map[logE].yaxis.min()
            ymax = list_eigen_xyoff_map[logE].yaxis.max()
            im = axbig.imshow(list_eigen_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    fig.savefig(f'output_plots/eigenmap_{source_name}_{input_epoch}_rank{rank}',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    figsize_x = 2.*logE_nbins
    figsize_y = 2.*gcut_bins
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    ax_idx = 0
    gs = GridSpec(gcut_bins, logE_nbins, hspace=0.1, wspace=0.1)
    for logE in range(0,logE_nbins):
        for gcut in range(0,gcut_bins):
            ax_idx = logE + (gcut-0)*logE_nbins + 1
            axbig = fig.add_subplot(gs[ax_idx-1])
            if logE==0:
                if gcut==0:
                    axbig.set_ylabel('SR')
                else:
                    axbig.set_ylabel(f'CR{gcut}')
            if gcut==0:
                axbig.set_title(f'{pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
            if not logE==0:
                axbig.axes.get_yaxis().set_visible(False)
            if not gcut==gcut_bins-1:
                axbig.axes.get_xaxis().set_visible(False)
            xmin = list_eigen_xyoff_map[logE].xaxis.min()
            xmax = list_eigen_xyoff_map[logE].xaxis.max()
            ymin = list_eigen_xyoff_map[logE].yaxis.min()
            ymax = list_eigen_xyoff_map[logE].yaxis.max()
            im = axbig.imshow(list_eigen_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    fig.savefig(f'output_plots/eigenmap_{source_name}_{input_epoch}_rank{rank}_tanspose',bbox_inches='tight')
    axbig.remove()


U_full, S_full, VT_full = np.linalg.svd(big_matrix_fullspec,full_matrices=False) # perform better for perturbation method
print (f'big_matrix_fullspec.shape = {big_matrix_fullspec.shape}')
print (f'S_full length = {len(S_full)}')

effective_matrix_rank = min(matrix_rank_fullspec,int(0.5*3./4.*(len(S_full)-1)))
print (f'effective_matrix_rank_fullspec = {effective_matrix_rank}')

U_eco = U_full[:, :effective_matrix_rank]
VT_eco = VT_full[:effective_matrix_rank, :]
S_eco = S_full[:effective_matrix_rank]
big_eigenvalues_fullspec = S_eco
big_eigenvectors_fullspec = VT_eco

sq_error_map_1d_fullspec = np.zeros_like(big_matrix_fullspec[0])
tolerance_map_1d_fullspec = np.zeros_like(big_matrix_fullspec[0])
norm_map_1d_fullspec = np.zeros_like(big_matrix_fullspec[0])
for run in range(0,len(big_matrix_fullspec)):
    data_xyoff_map_1d_fullspec = big_matrix_fullspec[run]
    truth_params = big_eigenvectors_fullspec @ data_xyoff_map_1d_fullspec
    eco_xyoff_map_1d_fullspec = big_eigenvectors_fullspec.T @ truth_params
    for pix in range(0,len(sq_error_map_1d_fullspec)):
        data = data_xyoff_map_1d_fullspec[pix]
        model = eco_xyoff_map_1d_fullspec[pix]
        sq_error_map_1d_fullspec[pix] += abs(data-model)
        norm_map_1d_fullspec[pix] += data
for pix in range(0,len(sq_error_map_1d_fullspec)):
    if sq_error_map_1d_fullspec[pix]==0.: continue
    tolerance_map_1d_fullspec[pix] = norm_map_1d_fullspec[pix]/sq_error_map_1d_fullspec[pix]

list_tolerance_xyoff_map = []
for logE in range(0,logE_nbins):
    tolerance_xyoff_map = MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)
    list_tolerance_xyoff_map += [tolerance_xyoff_map]
idx_1d = 0
for gcut in range(0,gcut_bins):
    for logE in range(0,logE_nbins):
        for idx_x in range(0,xoff_bins[logE]):
            for idx_y in range(0,yoff_bins[logE]):
                idx_1d += 1
                tolerance = tolerance_map_1d_fullspec[idx_1d-1]
                list_tolerance_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = tolerance
tolerance_max = np.max(tolerance_map_1d_fullspec)
fig.clf()
figsize_x = 2.*logE_nbins
figsize_y = 2.*gcut_bins
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
ax_idx = 0
gs = GridSpec(gcut_bins, logE_nbins, hspace=0.1, wspace=0.1)
for logE in range(0,logE_nbins):
    for gcut in range(0,gcut_bins):
        ax_idx = logE + (gcut-0)*logE_nbins + 1
        axbig = fig.add_subplot(gs[ax_idx-1])
        if logE==0:
            if gcut==0:
                axbig.set_ylabel('SR')
            else:
                axbig.set_ylabel(f'CR{gcut}')
        if gcut==0:
            axbig.set_title(f'{pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
        if not logE==0:
            axbig.axes.get_yaxis().set_visible(False)
        if not gcut==gcut_bins-1:
            axbig.axes.get_xaxis().set_visible(False)
        xmin = list_tolerance_xyoff_map[logE].xaxis.min()
        xmax = list_tolerance_xyoff_map[logE].xaxis.max()
        ymin = list_tolerance_xyoff_map[logE].yaxis.min()
        ymax = list_tolerance_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(list_tolerance_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=0.,vmax=tolerance_max,aspect='auto')
fig.savefig(f'output_plots/tolerance_map_{source_name}_{input_epoch}_tanspose',bbox_inches='tight')
axbig.remove()


rank_index = []
for entry in range(0,len(S_full)):
    rank_index += [entry+1]

plot_max_rank = min(int(0.5*len(S_full)),300)

fig.clf()
figsize_x = 6.4
figsize_y = 4.8
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)
axbig = fig.add_subplot()
label_x = '$k$'
label_y = '$\sigma_{k}$'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.set_xlim(1,plot_max_rank)
axbig.set_ylim(S_full[plot_max_rank-1],2.*S_full[0])
axbig.set_xscale('log')
axbig.set_yscale('log')
axbig.plot(rank_index,S_full)
fig.savefig(f'{smi_dir}/output_plots/signularvalue_{source_name}_{input_epoch}_fullspec.png',bbox_inches='tight')
axbig.remove()
    
max_matrix_rank = min(5,big_eigenvectors_fullspec.shape[0])
for rank in range(0,max_matrix_rank):

    list_eigen_xyoff_map = []
    for logE in range(0,logE_nbins):
        eigen_xyoff_map = MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)
        list_eigen_xyoff_map += [eigen_xyoff_map]

    idx_1d = 0
    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            for idx_x in range(0,xoff_bins[logE]):
                for idx_y in range(0,yoff_bins[logE]):
                    idx_1d += 1
                    list_eigen_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += big_eigenvectors_fullspec[rank][idx_1d-1]


    fig.clf()
    figsize_x = 2.*gcut_bins
    figsize_y = 2.*logE_nbins
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    ax_idx = 0
    for logE in range(0,logE_nbins):
        for gcut in range(0,gcut_bins):
            ax_idx = gcut + (logE-0)*gcut_bins + 1
            axbig = fig.add_subplot(logE_nbins,gcut_bins,ax_idx)
            if logE==0:
                if gcut==0:
                    axbig.set_title('SR')
                else:
                    axbig.set_title(f'CR{gcut}')
            if gcut==0:
                axbig.set_ylabel(f'E = {pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
            xmin = list_eigen_xyoff_map[logE].xaxis.min()
            xmax = list_eigen_xyoff_map[logE].xaxis.max()
            ymin = list_eigen_xyoff_map[logE].yaxis.min()
            ymax = list_eigen_xyoff_map[logE].yaxis.max()
            im = axbig.imshow(list_eigen_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    fig.savefig(f'output_plots/fullspec_eigenmap_{source_name}_{input_epoch}_rank{rank}',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    figsize_x = 2.*logE_nbins
    figsize_y = 2.*gcut_bins
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    ax_idx = 0
    gs = GridSpec(gcut_bins, logE_nbins, hspace=0.1, wspace=0.1)
    for logE in range(0,logE_nbins):
        for gcut in range(0,gcut_bins):
            ax_idx = logE + (gcut-0)*logE_nbins + 1
            axbig = fig.add_subplot(gs[ax_idx-1])
            if logE==0:
                if gcut==0:
                    axbig.set_ylabel('SR')
                else:
                    axbig.set_ylabel(f'CR{gcut}')
            if gcut==0:
                axbig.set_title(f'{pow(10.,logE_bins[logE]):0.2f}-{pow(10.,logE_bins[logE+1]):0.2f} TeV')
            if not logE==0:
                axbig.axes.get_yaxis().set_visible(False)
            if not gcut==gcut_bins-1:
                axbig.axes.get_xaxis().set_visible(False)
            xmin = list_eigen_xyoff_map[logE].xaxis.min()
            xmax = list_eigen_xyoff_map[logE].xaxis.max()
            ymin = list_eigen_xyoff_map[logE].yaxis.min()
            ymax = list_eigen_xyoff_map[logE].yaxis.max()
            im = axbig.imshow(list_eigen_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
    fig.savefig(f'output_plots/fullspec_eigenmap_{source_name}_{input_epoch}_rank{rank}_tanspose',bbox_inches='tight')
    axbig.remove()

output_filename = f'{smi_output}/eigenvectors_{source_name}_{onoff}_{input_epoch}_{sky_tag}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump([big_eigenvalues,big_eigenvectors,big_xyoff_map_1d,big_eigenvalues_fullspec,big_eigenvectors_fullspec,big_xyoff_map_1d_fullspec,tolerance_map_1d_fullspec], file)


print ('SVD eigenvectors saved.')
