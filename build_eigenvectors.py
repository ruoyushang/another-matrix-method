
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
matrix_rank = common_functions.matrix_rank
ReadOffRunListFromFile = common_functions.ReadOffRunListFromFile
build_big_camera_matrix = common_functions.build_big_camera_matrix

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_input = os.environ.get("SMI_INPUT")
smi_dir = os.environ.get("SMI_DIR")

print ('loading matrix pickle data... ')
input_filename = f'{smi_dir}/output_eigenvector/big_off_matrix.pkl'
big_matrix = pickle.load(open(input_filename, "rb"))

print ('Computing SVD eigenvectors...')
big_eigenvectors = []
for logE in range(0,logE_bins):
    U_full, S_full, VT_full = np.linalg.svd(big_matrix[logE],full_matrices=False)
    U_eco = U_full[:, :matrix_rank]
    VT_eco = VT_full[:matrix_rank, :]
    big_eigenvectors += [VT_eco]

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Rank'
    label_y = 'Signular value'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.set_xlim(0,10)
    axbig.plot(S_full)
    fig.savefig(f'{smi_dir}/output_plots/signularvalue_logE{logE}.png',bbox_inches='tight')
    axbig.remove()

output_filename = f'{smi_dir}/output_eigenvector/eigenvectors.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_eigenvectors, file)

print ('SVD eigenvectors saved.')
