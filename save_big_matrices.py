
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
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")

source_name = sys.argv[1]
input_epoch = sys.argv[2] # 'V4', 'V5' or 'V6'
print (f'source_name = {source_name}, input_epoch = {input_epoch}')

off_runlist = ReadOffRunListFromFile(f'/nevis/tehanu/home/ryshang/veritas_analysis/easy-matrix-method/output_vts_hours/PairList_{source_name}_{input_epoch}.txt')
print (off_runlist)
big_off_matrix = build_big_camera_matrix(smi_input,off_runlist,max_runs=1e10,is_on=False)

output_filename = f'{smi_output}/big_off_matrix_{source_name}_{input_epoch}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_off_matrix, file)


print ('Big matrices saved.')
