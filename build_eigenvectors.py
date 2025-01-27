
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
prepare_vector_for_least_square = common_functions.prepare_vector_for_least_square
convert_multivar_to_xyoff_vector1d = common_functions.convert_multivar_to_xyoff_vector1d
convert_multivar_to_xyvar_vector1d = common_functions.convert_multivar_to_xyvar_vector1d
weighted_least_square_solution = common_functions.weighted_least_square_solution
convert_xyoff_vector1d_to_map3d = common_functions.convert_xyoff_vector1d_to_map3d
sortFirst = common_functions.sortFirst

n_params = logE_nbins * gcut_bins

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
eigen_tag = os.environ.get("EIGEN_TAG")
bin_tag = os.environ.get("BIN_TAG")
cr_tag = os.environ.get("CR_TAG")
ana_dir = os.environ.get("ANA_DIR")

sky_tag = f"{cr_tag}_{bin_tag}_{eigen_tag}"

source_name = sys.argv[1]
onoff = sys.argv[2]
input_epoch = sys.argv[3] # 'V4', 'V5' or 'V6'

#print (f"line {cf.f_lineno}")

# Run the 'ls' command and capture its output
result = subprocess.run(['ls',f'{smi_output}/{ana_dir}'], capture_output=True, text=True)
# Split the output into a list by newlines
file_list = result.stdout.splitlines()

list_filename = []
total_matrix_fullspec = []
for k in range(0,len(file_list)):
    if f'big_off_matrix_{source_name}_{onoff}_{input_epoch}_{cr_tag}_{bin_tag}' in file_list[k]:
        input_filename = f'{smi_output}/{ana_dir}/{file_list[k]}'
        print (f'input_filename = {input_filename}')
        big_matrix_pkl = pickle.load(open(input_filename, "rb"))
        list_filename += [(big_matrix_pkl[0],input_filename)]
        total_matrix_fullspec += big_matrix_pkl[2]
print (f"len(total_matrix_fullspec) = {len(total_matrix_fullspec)}")

list_filename.sort(key=sortFirst)

big_exposure = []
big_matrix_fullspec = []

print ('loading matrix pickle data... ')
for k in range(0,len(list_filename)):
    if f'big_off_matrix_{source_name}_{onoff}_{input_epoch}_{cr_tag}_{bin_tag}' in list_filename[k][1]:
        input_filename = f'{list_filename[k][1]}'
        print (f'input_filename = {input_filename}')
        big_matrix_pkl = pickle.load(open(input_filename, "rb"))

        onrun_elev = big_matrix_pkl[0]
        list_offrun_expo = big_matrix_pkl[1]
        list_offrun_matrix = big_matrix_pkl[2]
        if len(list_offrun_matrix)==0:
            continue

        big_exposure += list_offrun_expo
        big_matrix_fullspec += list_offrun_matrix

        #sum_offrun_expo = 0.
        #sum_offrun_matrix = np.zeros_like(list_offrun_matrix[0])
        #for offrun in range(0,len(list_offrun_expo)):
        #    sum_offrun_expo += list_offrun_expo[offrun]
        #    sum_offrun_matrix += list_offrun_matrix[offrun]
        #big_exposure += [sum_offrun_expo]
        #big_matrix_fullspec += [sum_offrun_matrix]


big_matrix_fullspec = np.array(big_matrix_fullspec)

delete_entries = []
for entry in range(0,len(big_matrix_fullspec)):
    norm = np.sum(big_matrix_fullspec[entry])
    expo = big_exposure[entry]
    if expo<15./60. or norm<200.:
        delete_entries += [entry]

new_xyoff_matrix_fullspec = []
new_xyvar_matrix_fullspec = []
for entry in range(0,len(big_matrix_fullspec)):
    if entry in delete_entries: continue
    norm = big_exposure[entry]
    new_xyoff_matrix_fullspec += [convert_multivar_to_xyoff_vector1d(big_matrix_fullspec[entry]/norm)]
    new_xyvar_matrix_fullspec += [convert_multivar_to_xyvar_vector1d(big_matrix_fullspec[entry]/norm)]
new_xyoff_matrix_fullspec = np.array(new_xyoff_matrix_fullspec)
new_xyvar_matrix_fullspec = np.array(new_xyvar_matrix_fullspec)


# Train a neural net to predict SR normalization

list_sr_map_1d = [[] for entry in range(0,logE_nbins)]
list_sr_weight = [[] for entry in range(0,logE_nbins)]
list_cr_map_1d = [[] for entry in range(0,logE_nbins)]
for entry in range(0,len(big_matrix_fullspec)):
    if entry in delete_entries: continue
    norm = big_exposure[entry]
    sr_map_1d, cr_map_1d = prepare_vector_for_least_square(big_matrix_fullspec[entry]/norm)
    for logE in range(0,logE_nbins):
        list_sr_map_1d[logE] += [sr_map_1d[logE]/norm]
        list_sr_weight[logE] += [np.ones_like(sr_map_1d[logE])]
        list_cr_map_1d[logE] += [cr_map_1d/norm]
list_sr_map_1d = np.array(list_sr_map_1d)
list_sr_weight = np.array(list_sr_weight)
list_cr_map_1d = np.array(list_cr_map_1d)
print (f"list_sr_map_1d.shape = {list_sr_map_1d.shape}")
print (f"list_sr_weight.shape = {list_sr_weight.shape}")
print (f"list_cr_map_1d.shape = {list_cr_map_1d.shape}")


model = []
model_err = []
for logE in range(0,logE_nbins):
    logE_model, logE_model_err = weighted_least_square_solution(list_cr_map_1d[logE],list_sr_map_1d[logE],list_sr_weight[logE],plot_tag=f'{source_name}_{onoff}_{input_epoch}_{sky_tag}')
    model += [logE_model]
    model_err += [logE_model_err]
output_filename = f'{smi_output}/{ana_dir}/model_least_square_{source_name}_{onoff}_{input_epoch}_{sky_tag}.pkl'
with open(output_filename, "wb") as file:
    pickle.dump([model,model_err], file)

#output_filename = f'{smi_output}/{ana_dir}/sr_norm_model_{source_name}_{onoff}_{input_epoch}_{cr_tag}_{bin_tag}_{sky_tag}.pkl'
#make_neuralnet_model(list_cr_map_1d, list_sr_norm, output_filename)
        
        
print ('Computing SVD eigenvectors...')
        
avg_xyoff_map_1d_fullspec = np.zeros_like(new_xyoff_matrix_fullspec[0])
avg_xyvar_map_1d_fullspec = np.zeros_like(new_xyvar_matrix_fullspec[0])
for entry in range(0,len(new_xyoff_matrix_fullspec)):
    avg_xyoff_map_1d_fullspec += new_xyoff_matrix_fullspec[entry]
for entry in range(0,len(new_xyvar_matrix_fullspec)):
    avg_xyvar_map_1d_fullspec += new_xyvar_matrix_fullspec[entry]


avg_xyoff_map =  convert_xyoff_vector1d_to_map3d(avg_xyoff_map_1d_fullspec)
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
        xmin = avg_xyoff_map[logE].xaxis.min()
        xmax = avg_xyoff_map[logE].xaxis.max()
        ymin = avg_xyoff_map[logE].yaxis.min()
        ymax = avg_xyoff_map[logE].yaxis.max()
        wmin = avg_xyoff_map[logE].waxis[:,:,gcut].min()
        wmax = avg_xyoff_map[logE].waxis[:,:,gcut].max()
        zmax = max(abs(wmax),abs(wmin))
        im = axbig.imshow(avg_xyoff_map[logE].waxis[:,:,gcut].T,extent=(xmin,xmax,ymin,ymax),aspect='auto',origin='lower',cmap='coolwarm')
fig.savefig(f'output_plots/fullspec_avgmap_xyoff_{source_name}_{input_epoch}_tanspose',bbox_inches='tight')
axbig.remove()


def find_elbow_rank(S_vtr):

    elbow_rank = 1
    if S_vtr[0]==0.:
        return elbow_rank

    for entry in range(0,len(S_vtr)-1):
        ratio = S_vtr[entry]/S_vtr[entry+1]
        if ratio-1.<0.1:
            elbow_rank = entry + 1
            break

    return elbow_rank

print (f'new_xyoff_matrix_fullspec.shape = {new_xyoff_matrix_fullspec.shape}')
U_full, S_full, VT_full = np.linalg.svd(new_xyoff_matrix_fullspec,full_matrices=False) # perform better for perturbation method
effective_matrix_rank_fullspec = min(matrix_rank_fullspec,int(0.5*(len(S_full)-1)))
#elbow_rank = find_elbow_rank(S_full)
#effective_matrix_rank_fullspec = min(effective_matrix_rank_fullspec,elbow_rank)
print (f'effective_matrix_rank_fullspec = {effective_matrix_rank_fullspec}')
U_eco = U_full[:, :effective_matrix_rank_fullspec]
VT_eco = VT_full[:effective_matrix_rank_fullspec, :]
S_eco = S_full[:effective_matrix_rank_fullspec]
big_xyoff_eigenvalues_fullspec = S_eco
big_xyoff_eigenvectors_fullspec = VT_eco

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
fig.savefig(f'{smi_dir}/output_plots/xyoff_signularvalue_{source_name}_{input_epoch}.png',bbox_inches='tight')
axbig.remove()
    

max_matrix_rank = min(matrix_rank_fullspec,big_xyoff_eigenvectors_fullspec.shape[0])
for rank in range(0,max_matrix_rank):

    eigen_xyoff_map =  convert_xyoff_vector1d_to_map3d(big_xyoff_eigenvectors_fullspec[rank])

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
            xmin = eigen_xyoff_map[logE].xaxis.min()
            xmax = eigen_xyoff_map[logE].xaxis.max()
            ymin = eigen_xyoff_map[logE].yaxis.min()
            ymax = eigen_xyoff_map[logE].yaxis.max()
            wmin = eigen_xyoff_map[logE].waxis[:,:,gcut].min()
            wmax = eigen_xyoff_map[logE].waxis[:,:,gcut].max()
            zmax = max(abs(wmax),abs(wmin))
            im = axbig.imshow(eigen_xyoff_map[logE].waxis[:,:,gcut].T,extent=(xmin,xmax,ymin,ymax),vmin=-zmax,vmax=zmax,aspect='auto',origin='lower',cmap='coolwarm')
    fig.savefig(f'output_plots/fullspec_eigenmap_xyoff_{source_name}_{input_epoch}_rank{rank}_tanspose',bbox_inches='tight')
    axbig.remove()



output_filename = f'{smi_output}/{ana_dir}/model_eigenvectors_{source_name}_{onoff}_{input_epoch}_{sky_tag}.pkl'
with open(output_filename,"wb") as file:
    models = []
    models += [[big_xyoff_eigenvalues_fullspec,big_xyoff_eigenvectors_fullspec,avg_xyoff_map_1d_fullspec]]
    pickle.dump(models, file)


print ('SVD eigenvectors saved.')
