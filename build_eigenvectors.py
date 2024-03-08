
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
big_matrix = pickle.load(open(input_filename, "rb"))
input_filename = f'{smi_output}/big_off_matrix_ctl_{source_name}_{input_epoch}.pkl'
big_matrix_ctl = pickle.load(open(input_filename, "rb"))

print ('Computing SVD eigenvectors...')
U_full, S_full, VT_full = np.linalg.svd(big_matrix,full_matrices=False)
U_eco = U_full[:, :matrix_rank]
VT_eco = VT_full[:matrix_rank, :]
big_eigenvectors = VT_eco

u_full, s_full, vT_full = np.linalg.svd(big_matrix_ctl,full_matrices=False)
u_eco = u_full[:, :matrix_rank]
vT_eco = vT_full[:matrix_rank, :]
big_eigenvectors_ctl = vT_eco

fig.clf()
axbig = fig.add_subplot()
label_x = 'Rank'
label_y = 'Signular value'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.set_xlim(0,20)
axbig.set_yscale('log')
axbig.plot(S_full)
fig.savefig(f'{smi_dir}/output_plots/signularvalue_{source_name}_{input_epoch}.png',bbox_inches='tight')
axbig.remove()

output_filename = f'{smi_output}/eigenvectors_{source_name}_{input_epoch}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_eigenvectors, file)

output_filename = f'{smi_output}/eigenvectors_ctl_{source_name}_{input_epoch}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_eigenvectors_ctl, file)

print ('SVD eigenvectors saved.')
