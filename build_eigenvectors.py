
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.optimize import least_squares, minimize
from common_functions import MyArray1D
from common_functions import MyArray3D

import common_functions

logE_nbins = common_functions.logE_nbins
matrix_rank = common_functions.matrix_rank
ReadOffRunListFromFile = common_functions.ReadOffRunListFromFile
build_big_camera_matrix = common_functions.build_big_camera_matrix

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_input = os.environ.get("SMI_INPUT")
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")

source_name = sys.argv[1]
input_epoch = sys.argv[2] # 'V4', 'V5' or 'V6'

print ('loading matrix pickle data... ')
input_filename = f'{smi_output}/big_off_matrix_{source_name}_{input_epoch}.pkl'
print (f'input_filename = {input_filename}')
big_matrix = pickle.load(open(input_filename, "rb"))

#input_filename = f'{smi_output}/big_off_matrix_ctl_{source_name}_{input_epoch}.pkl'
#big_matrix_ctl = pickle.load(open(input_filename, "rb"))

print ('Computing SVD eigenvectors...')
min_rank = 3
big_xyoff_map_1d = []
big_eigenvalues = []
big_eigenvectors = []

for logE in range(0,logE_nbins):
    big_xyoff_map_1d += [np.zeros_like(big_matrix[logE][0])]
for logE in range(0,logE_nbins):
    for entry in range(0,len(big_matrix[logE])):
        for pix in range(0,len(big_matrix[logE][entry])):
            big_xyoff_map_1d[logE][pix] += big_matrix[logE][entry][pix]

for logE in range(0,logE_nbins):
    U_full, S_full, VT_full = np.linalg.svd(big_matrix[logE],full_matrices=False)
    print (f'S_full length = {len(S_full)}')
    effective_matrix_rank = min(min_rank,int(0.5*(len(S_full)-1)))
    print (f'effective_matrix_rank = {effective_matrix_rank}')
    U_eco = U_full[:, :effective_matrix_rank]
    VT_eco = VT_full[:effective_matrix_rank, :]
    S_eco = S_full[:effective_matrix_rank]
    big_eigenvalues += [S_eco]
    big_eigenvectors += [VT_eco]

    rank_index = []
    for entry in range(0,len(S_full)):
        rank_index += [entry+1]
    
    plot_max_rank = min(int(0.5*len(S_full)),300)
    
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Rank'
    label_y = 'Signular value'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.set_xlim(1,plot_max_rank)
    axbig.set_ylim(S_full[plot_max_rank-1],2.*S_full[0])
    axbig.set_xscale('log')
    axbig.set_yscale('log')
    axbig.plot(rank_index,S_full)
    fig.savefig(f'{smi_dir}/output_plots/signularvalue_{source_name}_{input_epoch}_logE{logE}.png',bbox_inches='tight')
    axbig.remove()


#U_full, S_full, VT_full = np.linalg.svd(big_matrix,full_matrices=False)
#print (f'S_full length = {len(S_full)}')
#effective_matrix_rank = min(min_rank,int(0.5*(len(S_full)-1)))
#print (f'effective_matrix_rank = {effective_matrix_rank}')
#U_eco = U_full[:, :effective_matrix_rank]
#VT_eco = VT_full[:effective_matrix_rank, :]
#S_eco = S_full[:effective_matrix_rank]
#big_eigenvalues = S_eco
#big_eigenvectors = VT_eco

#rank_index = []
#for entry in range(0,len(S_full)):
#    rank_index += [entry+1]
#
#plot_max_rank = min(int(0.5*len(S_full)),300)
#
#fig.clf()
#axbig = fig.add_subplot()
#label_x = 'Rank'
#label_y = 'Signular value'
#axbig.set_xlabel(label_x)
#axbig.set_ylabel(label_y)
#axbig.set_xlim(1,plot_max_rank)
#axbig.set_ylim(S_full[plot_max_rank-1],2.*S_full[0])
#axbig.set_xscale('log')
#axbig.set_yscale('log')
#axbig.plot(rank_index,S_full)
#fig.savefig(f'{smi_dir}/output_plots/signularvalue_{source_name}_{input_epoch}.png',bbox_inches='tight')
#axbig.remove()


output_filename = f'{smi_output}/eigenvectors_{source_name}_{input_epoch}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump([big_eigenvalues,big_eigenvectors,big_xyoff_map_1d], file)


print ('SVD eigenvectors saved.')
