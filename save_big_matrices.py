
import os, sys
import ROOT
import numpy as np
import pickle

import common_functions

logE_bins = common_functions.logE_bins
matrix_rank = common_functions.matrix_rank
ReadOffRunListFromFile = common_functions.ReadOffRunListFromFile
ReadRunListFromFile = common_functions.ReadRunListFromFile
build_big_camera_matrix = common_functions.build_big_camera_matrix

smi_input = os.environ.get("SMI_INPUT")
smi_dir = os.environ.get("SMI_DIR")

off_runlist = ReadOffRunListFromFile('../easy-matrix-method/output_vts_hours/PairList_CrabNebula_elev_60_70_V6.txt')
print (off_runlist)
#big_off_matrix = build_big_camera_matrix(smi_input,off_runlist,max_runs=30,is_on=False)
big_off_matrix = build_big_camera_matrix(smi_input,off_runlist,max_runs=1e10,is_on=False)

output_filename = f'{smi_dir}/output_eigenvector/big_off_matrix.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_off_matrix, file)

#on_runlist = ReadRunListFromFile('../easy-matrix-method/output_vts_hours/RunList_CrabNebula_elev_60_70_V6.txt')
#big_on_matrix = build_big_camera_matrix(smi_input,on_runlist,max_runs=1e10,is_on=False)
#
#output_filename = f'{smi_dir}/output_eigenvector/big_on_matrix.pkl'
#with open(output_filename,"wb") as file:
#    pickle.dump(big_on_matrix, file)

print ('Big matrices saved.')
