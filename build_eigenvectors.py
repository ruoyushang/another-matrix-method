
import subprocess
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares, minimize
from inspect import currentframe, getframeinfo

cf = currentframe()
print (f"line {cf.f_lineno}")

import common_functions
from common_functions import MyArray1D
from common_functions import MyArray3D
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
cosmic_ray_like_chi2_fullspec = common_functions.cosmic_ray_like_chi2_fullspec
significance_li_and_ma = common_functions.significance_li_and_ma
prepare_vector_for_neuralnet = common_functions.prepare_vector_for_neuralnet
weighted_least_square_solution = common_functions.weighted_least_square_solution

print (f"line {cf.f_lineno}")

#import pytorch_functions
#make_neuralnet_model = pytorch_functions.make_neuralnet_model

print (f"line {cf.f_lineno}")

fig, ax = plt.subplots()
figsize_x = 6.4
figsize_y = 4.6
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_input = os.environ.get("SMI_INPUT")
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")
sky_tag = os.environ.get("SKY_TAG")
bin_tag = os.environ.get("BIN_TAG")

source_name = sys.argv[1]
onoff = sys.argv[2]
input_epoch = sys.argv[3] # 'V4', 'V5' or 'V6'

#print (f"line {cf.f_lineno}")

# Run the 'ls' command and capture its output
result = subprocess.run(['ls',f'{smi_output}'], capture_output=True, text=True)
# Split the output into a list by newlines
file_list = result.stdout.splitlines()

big_exposure = []
big_matrix_fullspec = []

print ('loading matrix pickle data... ')
for file in file_list:
    if f'big_off_matrix_{source_name}_{onoff}_{input_epoch}_{bin_tag}' in file:
        input_filename = f'{smi_output}/{file}'
        print (f'input_filename = {input_filename}')
        big_matrix_pkl = pickle.load(open(input_filename, "rb"))
        big_exposure += big_matrix_pkl[0]
        big_matrix_fullspec += big_matrix_pkl[1]

big_matrix_fullspec = np.array(big_matrix_fullspec)

delete_entries = []
for entry in range(0,len(big_matrix_fullspec)):
    norm = np.sum(big_matrix_fullspec[entry])
    if norm<1000.:
        delete_entries += [entry]

new_matrix_fullspec = []
for entry in range(0,len(big_matrix_fullspec)):
    if entry in delete_entries: continue
    norm = big_exposure[entry]
    new_matrix_fullspec += [big_matrix_fullspec[entry]/norm]
new_matrix_fullspec = np.array(new_matrix_fullspec)




# Train a neural net to predict SR normalization


list_sr_norm = []
list_sr_weight = []
list_cr_map_1d = []
for entry in range(0,len(big_matrix_fullspec)):
    if entry in delete_entries: continue
    norm = big_exposure[entry]
    sr_norm, cr_map_1d = prepare_vector_for_neuralnet(big_matrix_fullspec[entry]/norm)
    cr_norm = np.sum(cr_map_1d)
    #list_sr_norm += [sr_norm/cr_norm]
    #list_sr_weight += [[1. for logE in range(0,logE_nbins)]]
    #list_cr_map_1d += [cr_map_1d/cr_norm]
    list_sr_norm += [sr_norm]
    list_sr_weight += [[1. for logE in range(0,logE_nbins)]]
    list_cr_map_1d += [cr_map_1d]
list_sr_norm = np.array(list_sr_norm)
list_sr_weight = np.array(list_sr_weight)
list_cr_map_1d = np.array(list_cr_map_1d)
print (f"list_sr_norm.shape = {list_sr_norm.shape}")
print (f"list_sr_weight.shape = {list_sr_weight.shape}")
print (f"list_cr_map_1d.shape = {list_cr_map_1d.shape}")

model = []
model_err = []
for logE in range(0,logE_nbins):
    A, A_err = weighted_least_square_solution(list_cr_map_1d,list_sr_norm.T[logE],list_sr_weight.T[logE],plot_tag=f'{source_name}_{onoff}_{input_epoch}_{bin_tag}_{sky_tag}_logE{logE}')
    model += [A]
    model_err += [A_err]
output_filename = f'{smi_output}/sr_norm_model_{source_name}_{onoff}_{input_epoch}_{bin_tag}_{sky_tag}.pkl'
with open(output_filename, "wb") as file:
    pickle.dump([model,model_err], file)

#output_filename = f'{smi_output}/sr_norm_model_{source_name}_{onoff}_{input_epoch}_{bin_tag}_{sky_tag}.pkl'
#make_neuralnet_model(list_cr_map_1d, list_sr_norm, output_filename)



print ('Computing SVD eigenvectors...')

big_xyoff_map_1d_fullspec = np.zeros_like(new_matrix_fullspec[0])

for entry in range(0,len(new_matrix_fullspec)):
    for pix in range(0,len(new_matrix_fullspec[entry])):
        big_xyoff_map_1d_fullspec[pix] += new_matrix_fullspec[entry][pix]

U_full, S_full, VT_full = np.linalg.svd(new_matrix_fullspec,full_matrices=False) # perform better for perturbation method
print (f'new_matrix_fullspec.shape = {new_matrix_fullspec.shape}')
print (f'S_full length = {len(S_full)}')

effective_matrix_rank_fullspec = min(matrix_rank_fullspec,int(0.5*(len(S_full)-1)))
print (f'effective_matrix_rank_fullspec = {effective_matrix_rank_fullspec}')

U_eco = U_full[:, :effective_matrix_rank_fullspec]
VT_eco = VT_full[:effective_matrix_rank_fullspec, :]
S_eco = S_full[:effective_matrix_rank_fullspec]
big_eigenvalues_fullspec = S_eco
big_eigenvectors_fullspec = VT_eco


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
label_y = '$\\sigma_{k}$'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.set_xlim(1,plot_max_rank)
axbig.set_ylim(S_full[plot_max_rank-1],2.*S_full[0])
axbig.set_xscale('log')
axbig.set_yscale('log')
axbig.plot(rank_index,S_full)
fig.savefig(f'{smi_dir}/output_plots/signularvalue_{source_name}_{input_epoch}_fullspec.png',bbox_inches='tight')
axbig.remove()
    
max_matrix_rank = min(matrix_rank_fullspec,big_eigenvectors_fullspec.shape[0])
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

    for gcut in range(0,gcut_bins):
        for logE in range(0,logE_nbins):
            if np.sum(list_eigen_xyoff_map[logE].waxis[:,:,gcut])<0.:
                for idx_x in range(0,xoff_bins[logE]):
                    for idx_y in range(0,yoff_bins[logE]):
                        list_eigen_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = -1. * list_eigen_xyoff_map[logE].waxis[idx_x,idx_y,gcut]


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

output_filename = f'{smi_output}/eigenvectors_{source_name}_{onoff}_{input_epoch}_{bin_tag}_{sky_tag}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump([big_eigenvalues_fullspec,big_eigenvectors_fullspec,big_xyoff_map_1d_fullspec], file)


print ('SVD eigenvectors saved.')
